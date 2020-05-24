import json
import os
import random
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from utils import *


def image_transform(
        image_size: Union[int, List[int]],
        augmentation: dict = {},
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    cnt = 0

    # data augmentations
    horizontal_flip = augmentation.get('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1
        cnt += 1

    vertical_flip = augmentation.get('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1
        cnt += 1

    random_crop = augmentation.get('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)
        cnt += 1

    center_crop = augmentation.get('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))
        cnt += 1

    if len(augmentation) > cnt:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))

    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0],
                                                                                               **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]

    return transforms.Compose([v for v in t if v is not None])


def proposals_transform(
        image_size: Union[int, List[int]],
        augmentation: dict = {},
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """proposals transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    cnt = 0

    # data augmentations
    horizontal_flip = augmentation.get('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1
        cnt += 1

    vertical_flip = augmentation.get('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1
        cnt += 1

    random_crop = augmentation.get('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)
        cnt += 1

    center_crop = augmentation.get('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))
        cnt += 1

    if len(augmentation) > cnt:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))

    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0],
                                                                                               **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor()]

    return transforms.Compose([v for v in t if v is not None])


def get_dataloader(dataset: Callable[[str], Dataset],
                   num_workers: int = 0,
                   pin_memory: bool = True,
                   drop_last: bool = False,
                   # train_shuffle: bool = True,
                   train_shuffle: bool = False,
                   test_shuffle: bool = False,
                   train_augmentation: dict = {},
                   test_augmentation: dict = {},
                   batch_size: int = 1,
                   test_batch_size: Optional[int] = None) -> Tuple[
    List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """Return data loader list.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=train_shuffle)


class TrainDataset(Dataset):
    __meta_class__ = ABCMeta

    def __init__(self, data_dir, split, classes, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, 'JPEGImages')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, 'ImageSets', 'Main')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.image_labels = self._read_annotations(self.split)

        @abstractmethod
        def _read_annotation():
            pass


class TrainPRMDataset(TrainDataset):

    def __init__(self, data_dir, split, classes, transform=None, target_transform=None):
        super(TrainPRMDataset, self).__init__(data_dir, split, classes, transform, target_transform)

    def _read_annotations(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.data_dir))

        return list(class_labels.items())

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.image_labels)


class TrainFillingDataset(TrainDataset):

    def __init__(self, data_dir, split, classes, transform=None, target_transform=None):
        super(TrainFillingDataset, self).__init__(data_dir, split, classes, transform, target_transform)
        self.proposal_path = os.path.join(self.data_dir, 'ImageProposals')
        assert os.path.isdir(self.gt_path), 'Could not find image proposal folder "%s".' % self.proposal_path

    def _read_annotations(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.data_dir))

        return list(class_labels.items())

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        img_filename = os.path.join(self.image_dir, filename + '.jpg')
        proposal_filename = os.path.join(self.proposal_path, filename + '.json')
        img = Image.open(img_filename).convert('RGB')
        with open(proposal_filename, 'r') as f:
            proposals = list(map(rle_decode, json.load(f)))

        if self.transform is not None:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            img = self.transform(img)
            for idx in range(len(proposals)):
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                proposals[idx] = self.target_transform(Image.fromarray(proposals[idx] * 255))
            proposals = torch.cat(proposals)

        return img, proposals

    def __len__(self):
        return len(self.image_labels)


def train_dataset(
        train_type: str,
        split: str,
        data_dir: str,
        categories: List,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        delay_resolve=True) -> Dataset:
    if train_type == 'prm':
        return TrainPRMDataset(data_dir, split, categories, transform, target_transform)
    elif train_type == 'filling':
        return TrainFillingDataset(data_dir, split, categories, transform, target_transform)


class TestDataset(Dataset):

    def __init__(self, data_dir, split, image_size, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, 'JPEGImages')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.transform = transform
        self.img_name = self._read_img_name(split)
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _read_img_name(self, split):
        img_name = []
        path = os.path.join(self.data_dir, 'ImageSets', 'Main', split + '.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    filename = line.strip()
                    img_name.append(filename)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.data_dir))
        return img_name

    def __getitem__(self, index):
        filename = self.img_name[index]
        img_filename = os.path.join(self.image_dir, filename + '.jpg')
        img = Image.open(img_filename).convert('RGB')
        tensor_img = self.to_tensor(img)
        transform_img = self.transform(img) if self.transform else tensor_img
        return transform_img, tensor_img

    def __len__(self):
        return len(self.img_name)


def test_dataset(
        split: str,
        data_dir: str,
        image_size: int,
        transform: Optional[Callable] = None) -> Dataset:
    return TestDataset(data_dir, split, image_size, transform)
