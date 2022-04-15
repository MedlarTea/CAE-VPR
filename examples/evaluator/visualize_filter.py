import torchvision.models as models
import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
layer = list(model.features.children())
model = nn.Sequential(*layer)
model.load_state_dict(torch.load("/home/hanjing/Models/Auto-VPR/logs/vgg16_netvlad.pth"))
# Print layers and corresponding indicies

conv1_2 = model[2]
conv1_2_filters = [17, 33, 34, 57]

conv2_1 = model[5]
conv2_1_filters = [27, 40, 68, 73]

conv3_1 = model[10]
conv3_1_filters = [31, 61, 147, 182]

conv4_1 = model[17]
conv4_1_filters = [238, 251, 338, 495]

conv5_1 = model[24]
conv5_1_filters = [45, 271, 363, 409]

g_ascent = GradientAscent(model)

g_ascent.visualize(conv1_2, filter_idxs=[57, 36, 26, 32], title='conv1_2', save_path="./results/conv1_2_.png")
g_ascent.visualize(conv2_1, filter_idxs=[28, 73, 10, 14], title='conv2_1', save_path="./results/conv2_1_.png")
g_ascent.visualize(conv3_1, filter_idxs=[14, 22, 224, 138], title='conv3_1', save_path="./results/conv3_1_.png")
g_ascent.visualize(conv4_1, filter_idxs=[379, 368, 224, 388], title='conv4_1', save_path="./results/conv4_1_.png")
g_ascent.visualize(conv5_1, filter_idxs=[33, 290, 55, 39], title='conv5_1', save_path="./results/conv5_1_.png")