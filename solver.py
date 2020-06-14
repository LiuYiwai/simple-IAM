import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from PIL import Image
from torchvision import transforms

from losses import binary_cross_entropy_loss
from losses import multilabel_soft_margin_loss
from model import fc_resnet50
from model import finetune
from model import instance_extent_filling
from model import peak_response_mapping
from optims import sgd_optimizer


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""
        self.image_size = config['image_size']
        self.class_num = config['class_num']
        self.class_names = config['class_names']
        self.k_proposals = config['k_proposals']
        self.balance_factor = config['balance_factor']
        self.out_img_path = config['out_img_path']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.basebone = fc_resnet50(self.class_num, True)
        self.prm_module = peak_response_mapping(self.basebone, **config['model'])
        self.filling_module = instance_extent_filling(config)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.prm_module.to(self.device)
            self.filling_module.to(self.device)

        self.prm_module_criterion = multilabel_soft_margin_loss
        self.filling_module_criterion = binary_cross_entropy_loss

        self.max_epoch = config['max_epoch']

        self.params = finetune(self.prm_module, **config['finetune'])
        self.optimizer_prm = sgd_optimizer(self.params, **config['optimizer'])
        # self.optimizer_filling = sgd_optimizer(self.filling_module.parameters(), **config['optimizer'])
        self.optimizer_filling = torch.optim.Adam(self.filling_module.parameters(), lr=config['optimizer']['lr'])

        self.prm_epoch_offset = 0
        self.filling_epoch_offset = 0
        self.train_prm_resume = config['train_prm_resume']
        self.train_filling_resume = config['train_filling_resume']

        self.lr_update_step = 999999
        self.lr = config['optimizer']['lr']
        self.snapshot = config['snapshot']

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_prm_model(self):
        """Restore the trained generator and discriminator."""
        print('Loading the trained prm models')
        model_path = os.path.join(self.snapshot, 'model_prm_latest.pth.tar')
        if self.cuda:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
        self.prm_epoch_offset = checkpoint['epoch']
        self.prm_module.load_state_dict(checkpoint['state_dict'], False)
        self.lr = checkpoint['lr']

    def restore_filling_model(self):
        """Restore the trained generator and discriminator."""
        print('Loading the trained filling models')
        model_path = os.path.join(self.snapshot, 'model_filling_latest.pth.tar')
        if self.cuda:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
        self.filling_epoch_offset = checkpoint['epoch']
        self.filling_module.load_state_dict(checkpoint['state_dict'], False)
        self.lr = checkpoint['lr']

    def update_lr(self, lr):
        for param_group in self.optimizer_prm.param_groups:
            param_group['lr'] = lr

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(self, state, path, prefix, epoch, filename='checkpoint_prm.pth.tar'):
        prefix_save = os.path.join(path, prefix)
        # name = '%s_%d_%s' % (prefix_save, epoch, filename)
        # torch.save(state, name)
        # shutil.copyfile(name, '%s_latest.pth.tar' % prefix_save)
        torch.save(state, '%s_latest.pth.tar' % prefix_save)

    def discard_proposals(self, instance_list, threshold=0.2):
        selected_instances = []
        if len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou > threshold:
                    return x
                else:
                    return None

            instance_list = list(filter(iou_filter, instance_list))
            selected_instances.extend(instance_list)
        return selected_instances

    def pseudo_gt_sampling(self, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        discard_threshold = retrieval_cfg.get('discard_threshold', 0.2)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        pseudo_gt_mask = []

        for i in range(len(peak_response_maps)):
            batch_idx = peak_list[i, 0]
            class_idx = peak_list[i, 1]

            peak_response_map = peak_response_maps[i]

            # # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                balance_factor = param[0]
            elif isinstance(param, list):
                # independent params between classes
                balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            # select proposal
            proposal_val_list = []

            for j in range(min(proposal_count, len(proposals[batch_idx]))):
                raw_mask = np.array(
                    Image.fromarray(np.uint8(proposals[batch_idx][j])).resize(size=peak_response_map.shape,
                                                                              resample=Image.NEAREST))
                mask = raw_mask.astype(bool)

                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)

                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum()
                    proposal_val_list.append((val, class_idx, mask, peak_response_map))

            proposal_val_list = sorted(proposal_val_list, key=lambda x: x[0], reverse=True)
            candidates_num = min(self.k_proposals, len(proposal_val_list))
            proposal_val_list = proposal_val_list[0:candidates_num]

            if discard_threshold is not None:
                proposal_val_list = self.discard_proposals(proposal_val_list,
                                                           discard_threshold)

            # randomly samples a proposal
            choice_index = np.random.choice(len(proposal_val_list), 1)[0]

            pseudo_gt_mask.append(proposal_val_list[choice_index][2].astype(float))

        return pseudo_gt_mask

    def train_prm(self, train_data_loader, train_logger, val_data_loader=None, val_logger=None):
        if self.train_prm_resume:
            self.restore_prm_model()
        # Start training.
        print('Start training prm...')
        since = time.time()

        min_val_loss = np.inf
        min_val_epoch = 0

        for epoch in range(self.max_epoch):

            self.prm_module.train()  # Set model to training mode

            average_loss = 0.
            for iteration, (inp, tar) in enumerate(train_data_loader):
                inp = inp.to(self.device)
                tar = tar.to(self.device)

                _output = self.prm_module(inp)

                loss = self.prm_module_criterion(_output, tar, difficult_samples=True)

                average_loss += loss.item()
                print('trainning loss at (epoch %d, iteration %d) = %4f' % (
                    epoch + self.prm_epoch_offset + 1, iteration, average_loss / (iteration + 1)))

                self.optimizer_prm.zero_grad()
                loss.backward()
                self.optimizer_prm.step()

                #################### LOGGING #############################
                # sry, i don't know how to use this
                lr = self.optimizer_prm.param_groups[0]['lr']
                train_logger.add_scalar('lr', lr, epoch)
                train_logger.add_scalar('loss', loss, epoch)

            is_save_checkpoint = True
            if val_data_loader is not None:
                val_loss = self.validation_prm(val_data_loader)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_epoch = epoch
                else:
                    is_save_checkpoint = False

                print(f'minn val loss at epoch {min_val_epoch + self.prm_epoch_offset + 1} = {min_val_loss}')
                print(f'val loss at epoch {epoch + self.prm_epoch_offset + 1} = {loss}')

            if is_save_checkpoint:
                self.save_checkpoint({'arch': 'iam',
                                      'lr': self.lr,
                                      'epoch': epoch + self.prm_epoch_offset,
                                      'state_dict': self.prm_module.state_dict(),
                                      'error': average_loss},
                                     self.snapshot, 'model_prm', epoch + self.prm_epoch_offset,
                                     'checkpoint_prm.pth.tar')

            print('training %d epoch,loss is %.4f' % (epoch + self.prm_epoch_offset + 1, average_loss))
            # TO-DO: modify learning rates.

        time_elapsed = time.time() - since
        print('train phrase completed in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))

    def train_filling(self, train_data_loader, train_logger, val_data_loader=None, val_logger=None):
        if self.train_filling_resume:
            self.restore_filling_model()
        # Start training.
        self.restore_prm_model()
        print('Start training filling...')
        since = time.time()

        self.prm_module.inference()
        self.filling_module.train()

        p_list = []
        peak_list_list = []
        peak_response_maps_list = []
        for epoch in range(self.max_epoch):
            average_loss = 0.
            for iteration, (inp, proposals) in enumerate(train_data_loader):

                inp = inp.to(self.device)

                p_list.clear()
                peak_list_list.clear()
                peak_response_maps_list.clear()
                for idx in range(inp.size(0)):
                    item = inp[idx].unsqueeze(0)
                    return_tuple = self.prm_module(item)

                    if return_tuple is None:
                        continue
                    else:
                        visual_cues, p2, p3, p4 = return_tuple
                        peak_list = visual_cues[2]
                        peak_response_maps = visual_cues[3]
                        peak_list[:, 0] = idx
                        p_list.append([p2, p3, p4])
                        peak_list_list.append(peak_list)
                        peak_response_maps_list.append(peak_response_maps)

                if len(peak_response_maps_list) == 0:
                    print('trainning pass epoch %d iteration %d' % (epoch + self.filling_epoch_offset + 1,
                                                                    iteration))
                    continue

                p2 = torch.cat([item[0] for item in p_list])
                p3 = torch.cat([item[1] for item in p_list])
                p4 = torch.cat([item[2] for item in p_list])
                peak_list = torch.cat(peak_list_list)
                peak_response_maps = torch.cat(peak_response_maps_list)

                retrieval_cfg = dict(proposals=proposals, param=(self.balance_factor,))
                pseudo_gt_mask = self.pseudo_gt_sampling(peak_list, peak_response_maps, retrieval_cfg)
                pseudo_gt_mask = torch.Tensor(pseudo_gt_mask)

                p2 = p2.to(self.device)
                p3 = p3.to(self.device)
                p4 = p4.to(self.device)
                peak_list = peak_list.to(self.device)
                pseudo_gt_mask = pseudo_gt_mask.to(self.device)
                peak_response_maps = peak_response_maps.to(self.device)

                # using p2 p3 p4 form feature map
                _output = self.filling_module(peak_response_maps, peak_list, p2, p3, p4)

                loss = self.filling_module_criterion(_output, pseudo_gt_mask)

                average_loss += loss.item()
                print('trainning loss at (epoch %d, iteration %d) = %4f' % (
                    epoch + self.filling_epoch_offset + 1, iteration, average_loss / (iteration + 1)))

                self.optimizer_filling.zero_grad()
                loss.backward()
                self.optimizer_filling.step()

                #################### LOGGING #############################
                lr = self.optimizer_filling.param_groups[0]['lr']
                train_logger.add_scalar('lr', lr, epoch)
                train_logger.add_scalar('loss', loss, epoch)

            self.save_checkpoint({'arch': 'iam',
                                  'lr': self.lr,
                                  'epoch': epoch + self.filling_epoch_offset,
                                  'state_dict': self.filling_module.state_dict(),
                                  'error': average_loss},
                                 self.snapshot, 'model_filling', epoch + self.filling_epoch_offset,
                                 'checkpoint_filling.pth.tar')

            print('training %d epoch,loss is %.4f' % (epoch + self.filling_epoch_offset + 1, average_loss))
            # TO-DO: modify learning rates.

        time_elapsed = time.time() - since
        print('train phrase completed in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))

    def dense_crf_2d(self, img, output_probs):
        # out 为H，*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），
        # seg_map - 假设为语义分割的 mask, hxw, np.array 形式.

        h = output_probs.shape[0]
        w = output_probs.shape[1]

        output_probs = np.expand_dims(output_probs, 0)
        output_probs = np.append(1 - output_probs, output_probs, axis=0)

        d = dcrf.DenseCRF2D(w, h, 2)
        #
        output_probs[output_probs == 0] = 1e-10
        U = -np.log(output_probs)
        U = U.reshape((2, -1))
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(U)

        # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

        Q = d.inference(5)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q

    def inference(self, test_data_loader):
        self.restore_prm_model()
        self.restore_filling_model()

        # visual cue extraction
        self.filling_module.eval()

        p_list = []
        aware_list = []
        peak_list_list = []
        peak_response_maps_list = []
        class_response_maps_list = []
        for iteration, (inp, rar_img) in enumerate(test_data_loader):

            # self.model.eval()
            # confidence = self.model(inp)
            # for idx in range(len(self.class_names)):
            #     if confidence.data[0, idx] > 0:
            #         print('[class_idx: %d] %s (%.2f)' % (idx, self.class_names[idx], confidence[0, idx]))

            self.prm_module.inference()

            inp = inp.to(self.device)

            p_list.clear()
            aware_list.clear()
            peak_list_list.clear()
            peak_response_maps_list.clear()
            class_response_maps_list.clear()
            for idx in range(inp.size(0)):
                item = inp[idx].unsqueeze(0)

                return_tuple = self.prm_module(item)

                if return_tuple is None:
                    print('class aware pass')
                    img = transforms.ToPILImage()(rar_img[idx]).convert('RGB')
                    img = img.resize(size=(self.image_size, self.image_size), resample=Image.BICUBIC)
                    plt.imshow(img)
                    plt.title('class aware pass')
                    # plt.show()
                    plt.close('all')
                    # TODO save
                    continue
                else:
                    aware_list.append(idx)
                    visual_cues, p2, p3, p4 = return_tuple
                    confidence, class_response_maps, peak_list, peak_response_maps = visual_cues
                    peak_list[:, 0] = idx
                    p_list.append([p2, p3, p4])
                    peak_list_list.append(peak_list)
                    peak_response_maps_list.append(peak_response_maps)
                    class_response_maps_list.append(class_response_maps)

            if len(aware_list) == 0:
                print('inference pass')
                continue

            p2 = torch.cat([item[0] for item in p_list])
            p3 = torch.cat([item[1] for item in p_list])
            p4 = torch.cat([item[2] for item in p_list])
            peak_list = torch.cat(peak_list_list)
            peak_response_maps = torch.cat(peak_response_maps_list)
            class_response_maps = torch.cat(class_response_maps_list)

            p2 = p2.to(self.device)
            p3 = p3.to(self.device)
            p4 = p4.to(self.device)
            peak_response_maps = peak_response_maps.to(self.device)

            with torch.no_grad():
                instance_activate_maps = self.filling_module(peak_response_maps, peak_list, p2, p3, p4)
            instance_activate_maps = instance_activate_maps.detach()

            if len(aware_list) == 0:
                print('No class peak response detected')
            else:
                # TODO No class peak save raw_img
                for it, batch_idx in enumerate(aware_list):
                    plt.figure(figsize=(5, 5))
                    class_idx = peak_list[it, 1]
                    mask = peak_list[:, 0] == batch_idx
                    num_plots = 2 + mask.sum().item() * 2
                    f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4), squeeze=False)
                    img = transforms.ToPILImage()(rar_img[batch_idx]).convert('RGB')
                    img = img.resize(size=(self.image_size, self.image_size), resample=Image.BICUBIC)
                    axarr[0][0].imshow(img)
                    axarr[0][0].set_title('Image')
                    axarr[0][0].axis('off')
                    axarr[0][1].imshow(class_response_maps[it, class_idx], interpolation='bicubic',
                                       cmap=plt.cm.gray)
                    axarr[0][1].set_title('Class Response Map')
                    axarr[0][1].axis('off')
                    raw_img = np.array(img)
                    raw_img = raw_img.astype(np.uint8)
                    for idx, (iam, peak) in enumerate(
                            sorted(zip(instance_activate_maps[mask], peak_list[mask]), key=lambda v: v[-1][-1])):
                        iam_img = iam.cpu().numpy()
                        crf_img = self.dense_crf_2d(raw_img, iam_img)
                        axarr[0][2 * idx + 2].imshow(iam_img,
                                                     cmap=plt.cm.gray)
                        axarr[0][2 * idx + 2].set_title(
                            'Instance Activation Map ("%s")' % (self.class_names[peak[1].item()]))
                        axarr[0][2 * idx + 2].axis('off')

                        axarr[0][2 * idx + 3].imshow(crf_img,
                                                     cmap=plt.cm.gray,
                                                     )
                        axarr[0][2 * idx + 3].set_title(
                            'Predict ("%s")' % (self.class_names[peak[1].item()]))
                        axarr[0][2 * idx + 3].axis('off')
                    path = os.path.join(self.out_img_path, f"{iteration + 1}_{batch_idx}.jpg")
                    plt.savefig(path, bbox_inches='tight')
                    # TODO save predict as real out name
                    # plt.show()
                    plt.close('all')

    def validation_prm(self, data_loader):
        self.prm_module.eval()
        val_loss = 0.
        epochs = 0
        for iteration, (inp, tar) in enumerate(data_loader):
            inp = inp.to(self.device)
            tar = tar.to(self.device)

            with torch.no_grad():
                _output = self.prm_module(inp)

            loss = self.prm_module_criterion(_output, tar, difficult_samples=True)
            val_loss += loss

            epochs = iteration

        val_loss /= (epochs + 1)
        return val_loss
