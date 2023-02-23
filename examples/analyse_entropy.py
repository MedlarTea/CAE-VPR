from __future__ import print_function, absolute_import
import os
import time
import argparse
import string
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.nn import Parameter
from sklearn.metrics import average_precision_score

import cv2
import _pickle as cPickle

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature_dim = 512
        vgg = torchvision.models.vgg16(pretrained=True)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
    def forward(self, x):
        # s1 = time.time()
        N,C,H,W = x.size()
        x = self.base(x)   
        # print("VGG inference: {:.3f}".format(time.time()-s1))
        return x

class convAuto(nn.Module):
    def __init__(self, d1,d2,dimension):
        super(convAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, d1, (4,4), stride=(1,1), padding=0),
            nn.BatchNorm2d(d1),
            nn.PReLU(),

            nn.Conv2d(d1, d2, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d2),
            nn.PReLU(),

            nn.Conv2d(d2, dimension, (5,3), stride=(2,2), padding=0),   # dimension x 4 x 8
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, d2, (5,3), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d2),
            nn.PReLU(),

            nn.ConvTranspose2d(d2, d1, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(d1),
            nn.PReLU(),

            nn.ConvTranspose2d(d1, 512, (4,4), stride=(1,1), padding=0), 
            nn.BatchNorm2d(512),
            nn.PReLU()
            # nn.Tanh()
        )
    def forward(self,x):
        # s1 = time.time()
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        # print("convAuto inference: {:.3f}".format(time.time()-s1))
        return x

class VggConvAuto(nn.Module):
    def __init__(self, base_model, convAuto_model, islayerNorm=False):
        super(VggConvAuto, self).__init__()
        self.base_model = base_model
        self.convAuto_model = convAuto_model
        self.islayerNorm = islayerNorm
        if self.islayerNorm:
            self.layernorm = nn.LayerNorm([512, 30, 40], elementwise_affine=False)

    def forward(self, x):
        features = self.base_model(x)
        if self.islayerNorm:
            features = self.layernorm(features)
        encoded = self.convAuto_model(features)
        # return encoded
        return F.normalize(encoded, p=2, dim=-1)
        # return encoded


def copy_state_dict(state_dict, model, strip=None, replace=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        # print(name)
        if replace is not None and name.find(replace[0]) != -1:
            name = name.replace(replace[0], replace[1])
        # print(name)
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)
    missing = set(tgt_state.keys()) - copied_names
    if ((len(missing) > 0)):
        print("missing keys in state_dict:", missing)
    return model

def get_transformer_test(height, width, tokyo=False):
    test_transformer = [T.Resize(max(height,width) if tokyo else (height, width)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                   std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])]
    return T.Compose(test_transformer)

def get_data():
    img1 = "/home/hanjing/Models/OpenIBL_forRobotcar/smallTestSet/imageRepeat.jpg"
    img2 = "/home/hanjing/Models/OpenIBL_forRobotcar/smallTestSet/imageOrigin.jpg"
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img_transformer = get_transformer_test(480, 640)
    return img_transformer(img1).unsqueeze(0), img_transformer(img2).unsqueeze(0)

def get_model(args):
    base_model = VGG()
    convAuto_model = convAuto(d1=args.d1, d2=args.d2, dimension=args.dimension)
    model = VggConvAuto(base_model, convAuto_model, islayerNorm=True)
    return model

def calculate_entropy_batch(descriptors, imgFilenames=None):
    """calculate the entropy of the descriptors
    Args:
        descriptors(torch.Tensor): with N x d, l2-normalized
        image_path(str): the path of the image
    Output:
        entropy of the descriptors
    """
    descriptors = descriptors.cpu().detach().numpy()
    descriptors = abs(descriptors)
    MAX = np.max(descriptors)
    MIN = np.min(descriptors)
    if imgFilenames is not None:
        for i in range(len(imgFilenames)):
            plt.figure(figsize=(14,1))
            cmap=sns.color_palette("flare", as_cmap=True)
            sns.heatmap([descriptors[i]], cmap=cmap, vmax=MAX, vmin=MIN, cbar=False, yticklabels=False, xticklabels=False)
            plt.savefig(os.path.join(os.path.dirname(imgFilenames[i]), os.path.splitext(os.path.basename(imgFilenames[i]))[0]+"_batch_entropy.png"),dpi=600)
            plt.close()

def calculate_entropy(descriptor, image_path=None):
    """calculate the entropy of the descriptor
    Args:
        descriptor(torch.Tensor): with 1 x d, l2-normalized
        image_path(str): the path of the image
    Output:
        entropy of the descriptor
    """
    # descriptor = descriptor.squeeze(0).cpu().detach().numpy()  # 1-d dataset
    descriptor = descriptor.cpu().detach().numpy()  # 2-d dataset
    descriptor = abs(descriptor)
    # mean and std
    mean = np.mean(descriptor)
    std = np.std(descriptor)

    # max-element
    # ap = sum(np.max(descriptor)-descriptor)

    # mean-element
    # mean = np.mean(descriptor)
    # descriptorT = descriptor[descriptor<mean]
    # ap = sum(mean-descriptorT)

    # ap = 0
    # threshold = np.arange(0, descriptor.max(), 0.001)
    # values = []

    # gradient
    # d1 = descriptor[1:]
    # print(d1.shape)
    # d2 = descriptor[:-1]
    # print(d2.shape)
    # ap = sum(abs(d1-d2))

    # for th in threshold:
    #     ap = ap + (descriptor > th).sum()
    # for th in threshold:
        # values.append((descriptor > th).sum())
    # descriptor = np.clip(descriptor, 0, 1)
    # entropy = -descriptor * np.log(descriptor)
    # ap = ap / (1024*threshold.shape[0])
    # print("Entropy is: {:.5f}".format(ap))
    if image_path is not None:
        # plt.bar(range(descriptor.shape[0]), abs(descriptor), color="blue")
        # plt.bar(range(descriptor.shape[0]), abs(descriptor), label="Repeated (d={},ap={:.4f})".format(1024, ap), color="blue")
        # plt.legend()
        plt.figure(figsize=(14,1))
        # cmap=sns.color_palette("icefire", as_cmap=True)
        cmap=sns.color_palette("flare", as_cmap=True)
        sns.heatmap(descriptor,cmap=cmap, cbar=False, yticklabels=False, xticklabels=False)
        print(image_path)
        print("u={:.4f},std={:.4f}".format(mean,std))
        # s.set(xlabel="u={:.4f},std={:.4f}".format(mean,std))
        plt.savefig(os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0]+"_entropy.png"),dpi=600)
        plt.close()
    # np.savetxt(os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0]+".txt"), descriptor)
    # return threshold, values

def store_desc(args, descriptor, image_path):
    """store the descriptor
    Args:
        descriptor(torch.Tensor): with 1 x d, l2-normalized
        image_path(str): the path of the image
    Output:
        save the descriptor as 
    """
    descriptor = descriptor.squeeze(0).cpu().detach().numpy()
    save_path = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0]+"_"+str(int(args.dimension*32))+"d.txt")
    np.savetxt(save_path, descriptor)

def get_files_from_directory(directory_path):
    """
    Args:
        directory_path(str): path that stored the images
    Output:
        files(List): absolute paths of all images
    """
    files = []
    for x in os.listdir(directory_path):
        if (x.endswith("png") or x.endswith("jpg")) and x.find("entropy")==-1:
            # print(x)
            files.append(os.path.join(directory_path, x))
    return files

def main():
    args = parser.parse_args()

    main_worker(args)

def main_worker(args):
    # print(args)

    # get data
    img_repeat, img_origin = get_data()

    # Create model
    model = get_model(args)
    model.cuda()

    # Load from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        copy_state_dict(checkpoint['state_dict'], model, strip=None, replace=["module.", ""])
        start_epoch = checkpoint['epoch']
        best_recall5 = checkpoint['best_recall5']
        print("=> Start epoch {}  best recall5 {:.1%}"
                .format(start_epoch, best_recall5))

    directory = args.images_directory
    files = get_files_from_directory(directory)
    print(directory)
    print("Total: {} files".format(len(files)))
    print("Dimension: {}".format(args.dimension))
    model.eval()
    img_transformer = get_transformer_test(480, 640)

    # one by one
    # with torch.no_grad():
    #     s1 = time.time()
    #     for path in files:
    #         image_rgb = Image.fromarray(cv2.imread(path))

    #         image = img_transformer(image_rgb).unsqueeze(0).cuda()
    #         desc = model(image)
    #         # store_desc(args, desc, path)
    #         calculate_entropy(desc, image_path=path)
    #         # plt.plot(th, values)
    #     # plt.savefig(os.path.join(os.path.dirname(files[0]), "entropy.png"), dpi=600)
    #     print("Average used time: {:.3f}s".format((time.time()-s1)/len(files)))
    
    # batch
    with torch.no_grad():
        descs = []
        for path in files:
            image_rgb = Image.fromarray(cv2.imread(path))
            image = img_transformer(image_rgb).unsqueeze(0).cuda()
            descs.append(model(image))
        descs = torch.cat(descs, 0)
        calculate_entropy_batch(descs, imgFilenames=files)
            
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    parser.add_argument('--height', type=int, default=480, help="input height")
    parser.add_argument('--width', type=int, default=640, help="input width")

    parser.add_argument('--d1', type=int, default=128)
    parser.add_argument('--d2', type=int, default=128)
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--resume', type=str, default='../logs/convAuto/robotcar/vgg/lr0.001-bs128-islayernormTrue-d1-128-d2-128-dimension1024/checkpoint49.pth.tar', metavar='PATH')
    parser.add_argument('--images_directory', type=str, default="")
    

    main()
