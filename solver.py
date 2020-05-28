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
        self.optimizer_filling = torch.optim.RMSprop(self.filling_module.parameters())
        # self.optimizer_filling = torch.optim.Adam(self.filling_module.parameters())
        # self.optimizer_filling = torch.optim.Adadelta(self.filling_module.parameters())

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

            choice_index = np.random.choice(len(proposal_val_list), 1)[0]

            pseudo_gt_mask.append(proposal_val_list[choice_index][2].astype(float))

        return pseudo_gt_mask

    def train_prm(self, train_data_loader, train_logger, val_data_loader=None, val_logger=None):
        # self.restore_prm_model()
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
                    epoch + 1, iteration, average_loss / (iteration + 1)))

                self.optimizer_prm.zero_grad()
                loss.backward()
                self.optimizer_prm.step()

                #################### LOGGING #############################
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

                print(f'minn val loss at epoch {min_val_epoch + 1} = {min_val_loss}')
                print(f'val loss at epoch {epoch + 1} = {loss}')

            if is_save_checkpoint:
                self.save_checkpoint({'arch': 'iam',
                                      'lr': self.lr,
                                      'epoch': epoch,
                                      'state_dict': self.prm_module.state_dict(),
                                      'error': average_loss},
                                     self.snapshot, 'model_prm', epoch, 'checkpoint_prm.pth.tar')

            print('training %d epoch,loss is %.4f' % (epoch + 1, average_loss))
            # TO-DO: modify learning rates.

        time_elapsed = time.time() - since
        print('train phrase completed in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))

    def train_filling(self, train_data_loader, train_logger, val_data_loader=None, val_logger=None):
        # self.restore_filling_model()
        # Start training.
        self.restore_prm_model()
        print('Start training filling...')
        since = time.time()

        self.prm_module.inference()
        self.filling_module.train()

        peak_response_maps_list = []
        peak_list_list = []
        p2_list = []
        p3_list = []
        p4_list = []
        for epoch in range(self.max_epoch):
            average_loss = 0.
            for iteration, (inp, proposals) in enumerate(train_data_loader):

                inp = inp.to(self.device)

                peak_response_maps_list.clear()
                peak_list_list.clear()
                p2_list.clear()
                p3_list.clear()
                p4_list.clear()
                for idx in range(inp.size(0)):
                    item = inp[idx].unsqueeze(0)
                    return_tuple = self.prm_module(item)

                    if return_tuple is None:
                        continue
                    else:
                        visual_cues, p2, p3, p4 = return_tuple
                        peak_response_maps = visual_cues[3]
                        peak_list = visual_cues[2]
                        peak_list[:, 0] = idx
                        peak_response_maps_list.append(peak_response_maps)
                        peak_list_list.append(peak_list)
                        p2_list.append(p2)
                        p3_list.append(p3)
                        p4_list.append(p4)

                if len(peak_response_maps_list) == 0:
                    print('trainning pass epoch %d iteration %d' % (epoch + 1, iteration))
                    continue

                peak_response_maps = torch.cat(peak_response_maps_list)
                peak_list = torch.cat(peak_list_list)
                p2 = torch.cat(p2_list)
                p3 = torch.cat(p3_list)
                p4 = torch.cat(p4_list)

                retrieval_cfg = dict(proposals=proposals, param=(self.balance_factor,))
                pseudo_gt_mask = self.pseudo_gt_sampling(peak_list, peak_response_maps, retrieval_cfg)
                pseudo_gt_mask = torch.Tensor(pseudo_gt_mask)

                p2 = p2.to(self.device)
                p3 = p3.to(self.device)
                p4 = p4.to(self.device)
                peak_response_maps = peak_response_maps.to(self.device)
                pseudo_gt_mask = pseudo_gt_mask.to(self.device)

                _output = self.filling_module(peak_response_maps, peak_list, p2, p3, p4)

                loss = self.filling_module_criterion(_output, pseudo_gt_mask)

                average_loss += loss.item()
                print('trainning loss at (epoch %d, iteration %d) = %4f' % (
                    epoch + 1, iteration, average_loss / (iteration + 1)))

                self.optimizer_filling.zero_grad()
                loss.backward()
                self.optimizer_filling.step()

                #################### LOGGING #############################
                lr = self.optimizer_filling.param_groups[0]['lr']
                train_logger.add_scalar('lr', lr, epoch)
                train_logger.add_scalar('loss', loss, epoch)

            self.save_checkpoint({'arch': 'iam',
                                  'lr': self.lr,
                                  'epoch': epoch,
                                  'state_dict': self.filling_module.state_dict(),
                                  'error': average_loss},
                                 self.snapshot, 'model_filling', epoch, 'checkpoint_filling.pth.tar')

            print('training %d epoch,loss is %.4f' % (epoch + 1, average_loss))
            # TO-DO: modify learning rates.

        time_elapsed = time.time() - since
        print('train phrase completed in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))

    def dense_crf_2d(self, img, output_probs):
        # img 为H，*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），
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

        # d.addPairwiseGaussian(sxy=20, compat=3)
        # d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

        # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

        # Q = d.inference(5)
        Q = d.inference(3)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q

    def inference(self, test_data_loader):
        self.restore_prm_model()
        self.restore_filling_model()

        # Visual cue extraction
        self.filling_module.eval()
        # self.filling_module.train()

        peak_response_maps_list = []
        peak_list_list = []
        p2_list = []
        p3_list = []
        p4_list = []
        class_response_maps_list = []
        aware_list = []
        for iteration, (inp, rar_img) in enumerate(test_data_loader):

            # self.model.eval()
            # confidence = self.model(inp)
            # for idx in range(len(self.class_names)):
            #     if confidence.data[0, idx] > 0:
            #         print('[class_idx: %d] %s (%.2f)' % (idx, self.class_names[idx], confidence[0, idx]))

            self.prm_module.inference()

            inp = inp.to(self.device)

            peak_response_maps_list.clear()
            peak_list_list.clear()
            p2_list.clear()
            p3_list.clear()
            p4_list.clear()
            class_response_maps_list.clear()
            aware_list.clear()
            for idx in range(inp.size(0)):
                item = inp[idx].unsqueeze(0)
                return_tuple = self.prm_module(item)

                if return_tuple is None:
                    print('class aware pass')
                    img = transforms.ToPILImage()(rar_img[idx]).convert('RGB')
                    img = img.resize(size=(self.image_size, self.image_size), resample=Image.BICUBIC)
                    plt.imshow(img)
                    plt.title('class aware pass')
                    plt.show()
                    continue
                else:
                    aware_list.append(idx)
                    visual_cues, p2, p3, p4 = return_tuple
                    confidence, class_response_maps, peak_list, peak_response_maps = visual_cues
                    peak_list[:, 0] = idx
                    peak_response_maps_list.append(peak_response_maps)
                    peak_list_list.append(peak_list)
                    p2_list.append(p2)
                    p3_list.append(p3)
                    p4_list.append(p4)
                    class_response_maps_list.append(class_response_maps)

            if len(aware_list) == 0:
                print('inference pass')
                continue

            peak_response_maps = torch.cat(peak_response_maps_list)
            peak_list = torch.cat(peak_list_list)
            p2 = torch.cat(p2_list)
            p3 = torch.cat(p3_list)
            p4 = torch.cat(p4_list)
            class_response_maps = torch.cat(class_response_maps_list)

            p2 = p2.to(self.device)
            p3 = p3.to(self.device)
            p4 = p4.to(self.device)
            peak_response_maps = peak_response_maps.to(self.device)

            instance_activate_maps = self.filling_module(peak_response_maps, peak_list, p2, p3, p4)
            instance_activate_maps = instance_activate_maps.detach()

            if visual_cues is None:
                print('No class peak response detected')
            else:
                # TODO No class peak save raw_img
                for it, batch_idx in enumerate(aware_list):
                    plt.figure(figsize=(5, 5))
                    class_idx = peak_list[batch_idx, 1]
                    mask = peak_list[:, 0] == batch_idx
                    num_plots = 2 + mask.sum().item()
                    f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))
                    img = transforms.ToPILImage()(rar_img[batch_idx]).convert('RGB')
                    img = img.resize(size=(self.image_size, self.image_size), resample=Image.BICUBIC)
                    axarr[0].imshow(img)
                    axarr[0].set_title('Image')
                    axarr[0].axis('off')
                    axarr[1].imshow(class_response_maps[it, class_idx].cpu(), interpolation='bicubic',
                                    cmap=plt.cm.gray)
                    axarr[1].set_title('Class Response Map ("%s")' % self.class_names[class_idx])
                    axarr[1].axis('off')
                    raw_img = np.array(img)
                    raw_img = raw_img.astype(np.uint8)
                    for idx, (iam, peak) in enumerate(
                            sorted(zip(instance_activate_maps[mask], peak_list[mask]), key=lambda v: v[-1][-1])):
                        iam_img = iam.cpu().numpy()
                        crf_img = self.dense_crf_2d(raw_img, iam_img)
                        axarr[idx + 2].imshow(crf_img,
                                              cmap=plt.cm.gray)
                        axarr[idx + 2].set_title('Instance Activation Map ("%s")' % (self.class_names[peak[1].item()]))
                        axarr[idx + 2].axis('off')
                    plt.show()
                    # TODO save

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
