import torch
from torch import nn
import torch.nn.functional as F

class convAuto(nn.Module):
    def __init__(self, dimension):
        super(convAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, (4,4), stride=(1,1), padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, 1024, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, dimension, (5,3), stride=(2,2), padding=0),   # dimension x 4 x 8
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 1024, (5,3), stride=(2,2), padding=0), 
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.ConvTranspose2d(1024, 256, (7,5), stride=(2,2), padding=0), 
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 512, (4,4), stride=(1,1), padding=0), 
            nn.BatchNorm2d(512),
            nn.PReLU()
            # nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        encoded = x
        encoded = encoded.view(x.size(0),-1)
        decoded = self.decoder(x)
        return encoded, decoded

class convAutoAlextnet(nn.Module):
    def __init__(self, dimension):
        super(convAutoAlextnet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 512, (4,4), stride=(1,1), padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.Conv2d(512, 1024, (5,3), stride=(2,2), padding=0),
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, dimension, (5,3), stride=(2,2), padding=0),   # dimension x 4 x 8
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 1024, (5,3), stride=(2,2), padding=0),
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.ConvTranspose2d(1024, 512, (5,3), stride=(2,2), padding=0),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.ConvTranspose2d(512, 256, (4,4), stride=(1,1), padding=0),
            nn.BatchNorm2d(256),
            nn.PReLU()
            # nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        encoded = x
        print(encoded.size())
        encoded = encoded.view(x.size(0),-1)
        decoded = self.decoder(x)
        print(decoded.size())

        return encoded, decoded


class fullyAuto(nn.Module):
    def __init__(self, netvlad_model, pca, vggConvAuto_model, inputD, outputD, concatType="concat", normType="l2Norm"):
        super(fullyAuto, self).__init__()
        # base model, and are pretrained
        self.base_model = netvlad_model.base_model
        self.net_vlad = netvlad_model.net_vlad
        self.pca = pca
        self.convAuto_model = vggConvAuto_model.convAuto_model
        if (self.vggConvAuto_model.islayerNorm ==True):
            self.layerNorm1 = self.vggConvAuto_model.layerNorm

        # information about this model
        self.inputD = inputD
        self.outputD = outputD
        self.concatType = concatType
        self.normType = normType
        self.init_params()
        if self.normType == "layerNorm":
            self.layerNorm2 = nn.LayerNorm(self.inputD, elementwise_affine = False)

        
        nums = [self.inputD, self.inputD/2, self.inputD/4, self.inputD/8, self.outputD]
        self.encoder = nn.Sequential(
            nn.Linear(nums[0], nums[1]),
            nn.BatchNorm1d(nums[1]),
            nn.PReLU(),

            nn.Linear(nums[1], nums[2]),
            nn.BatchNorm1d(nums[2]),
            nn.PReLU(),

            nn.Linear(nums[2], nums[3]),
            nn.BatchNorm1d(nums[3]),
            nn.PReLU(),

            nn.Linear(nums[3], nums[4]),
            #nn.BatchNorm1d(dimension),
            #nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nums[4], nums[3]),
            nn.BatchNorm1d(nums[3]),
            nn.PReLU(),

            nn.Linear(nums[3], nums[2]),
            nn.BatchNorm1d(nums[2]),
            nn.PReLU(),

            nn.Linear(nums[2], nums[1]),
            nn.BatchNorm1d(nums[1]),
            nn.PReLU(),

            nn.Linear(nums[1], nums[0]),
            nn.BatchNorm1d(nums[0]),
            #nn.ReLU(),
        )
        

    def init_params(self):
        if self.islayerNorm == False:
            models = [self.base_model, self.net_vlad, self.pca, self.convAuto_model]
        else:
            models = [self.base_model, self.net_vlad, self.pca, self.convAuto_model, self.layerNorm]
        # 冻结vgg参数
        for model in models:
            layers = list(model.children())
            for l in layers:
                for p in l.parameters():
                    p.requires_grad = False
        if self.concatType == "concat":
            self.inputD = 2*self.inputD            


    def forward(self,x):

        features = self.base_model(x)
        netvlad = self.net_vlad(features)
        netvlad = self.pca(netvlad)  # this have been processed by l2Norm

        conv_encoded, _ = self.convAuto_model(features)  # this have not been processed
        conv_encoded = F.normalize(conv_encoded, p=2, dim=-1)  # l2Norm

        # concatenate
        if self.concatType == "concat":
            fusion = torch.cat((netvlad, conv_encoded), dim=-1)
        elif self.concatType == "sum":
            fusion = torch.sum((netvlad, conv_encoded), dim=-1)
        
        # normalization
        if self.concatType == "l2Norm":
            fusion = F.normalize(fusion, p=2, dim=-1)  # l2Norm
        elif self.concatType == "layerNorm":
            fusion = self.layerNorm2(fusion)
        
        # fusion
        full_encoded = self.encoder(fusion)
        full_decoded = self.encoder(full_encoded)
        
        return fusion, encoded, decoded
