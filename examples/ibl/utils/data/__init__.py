from __future__ import absolute_import

import torchvision.transforms as T

from .dataset import Dataset
from .preprocessor import Preprocessor
from .loader import GaussianBlur
class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

# this is for matconvnet
def get_transformer_train(height, width):
    train_transformer = [T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                         T.Resize((height, width)),
                         T.ToTensor(),
                         T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(train_transformer)

# this is for matconvnet
def get_transformer_test(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                   std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(test_transformer)

# for pytorch pretrained convnet
def get_transformer_train_pytorch(height, width):
    train_transformer = [T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                         T.Resize((height, width)),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]
    return T.Compose(train_transformer)

def get_transformer_test_pytorch(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]
    return T.Compose(test_transformer)

# for pytorch pretrained convnet
def get_transformer_train_viT(height, width, mean, std):
    train_transformer = [T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                         T.Resize((height, width)),
                         T.ToTensor(),
                         T.Normalize(mean=mean,
                                     std=std)]
    return T.Compose(train_transformer)

def get_transformer_test_viT(height, width, mean, std, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=mean,
                                     std=std)]
    return T.Compose(test_transformer)

def get_transformer_train_robotcar(height, width):
    train_transformer = [T.ColorJitter(0.7, 0.7, 0.7, 0.5),
                         T.Resize((height, width)),
                         T.ToTensor(),
                         T.Normalize(mean=[0.4864, 0.4904, 0.5544],
                                     std=[0.2967, 0.3000, 0.3000])]
    return T.Compose(train_transformer)
def get_transformer_test_robotcar(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.4864, 0.4904, 0.5544],
                                     std=[0.2967, 0.3000, 0.3000])]
    return T.Compose(test_transformer)
# for Robotcar, RGBSD, 'datelist_for_PR.txt'
# mean=[0.4864, 0.4904, 0.5544, 0.2419, 0.2471]
# std=[0.2967, 0.3000, 0.3000, 0.2536, 0.2040]

def get_transformer_train_moco(aug_plus=False):
    # for mapillary, RGBD, with noNight
    normalize = T.Normalize(mean=[0.4158, 0.4519, 0.4601],
                            std=[0.2460, 0.2603, 0.2843])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            T.RandomResizedCrop(224, scale=(0.2, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            T.RandomResizedCrop(224, scale=(0.2, 1.)),
            T.RandomGrayscale(p=0.2),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ]
    return T.Compose(augmentation)
    