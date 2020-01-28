import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, std=0.001)
        nn.init.constant(m.bias.data, 0.0)

class EmbeddingResnet(nn.Module):
    """
        base network for encoding images into embedding vector
    """
    def __init__(self, class_num, *args, **kwargs):
        super(EmbeddingResnet, self).__init__()
        
        # Pretrained Resnet
        resnet = models.resnet50(pretrained=True)
        backbone = nn.Sequential(resnet.conv1, 
                                resnet.bn1, 
                                resnet.relu, 
                                resnet.maxpool, 
                                resnet.layer1, 
                                resnet.layer2, 
                                resnet.layer3, 
                                resnet.layer4,
                                nn.AdaptiveMaxPool2d((1,1))
        )

        self.backbone = backbone

        fc_layer = nn.Sequential(nn.BatchNorm1d(2048))
        fc_layer.apply(weights_init)
        self.fc_layer = fc_layer

        # Freezing 1 -> 6 layers, Train only the last few layers
        count = 0
        for child in self.backbone.children():
            count += 1
            if count < 7:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x) # Backbone Resnet -> features
        x = torch.squeeze(x)
        fc = self.fc_layer(x)
        fc = F.normalize(fc, p=2, dim=1)
        return fc

"""
Common Params:
    z_dim: latent dimension
    gf_dim: generator fc dimension
    df_dim: discriminator fc dimension
    color_dim: color channels
    attr_range: range value of each attribute (0: No, 1: Yes, 2: Young, etc.)
                e.g Binary value --> attr_range = 2

    ------------- pyTorch sample Params ---------------------
    # torch.nn.ConvTranspose2d(in_channels, out_channels, 
    # 						kernel_size, stride=1, padding=0, 
    # 						output_padding=0, groups=1, 
    # 						bias=True, dilation=1, 
    # 						padding_mode='zeros')
"""
# Upsampling
class Generator(nn.Module):
    def __init__(self, z_dim=100, gf_dim=64, color_dim=3, 
                attr_range=4, n_attr=27, attr_size=4):

        super(Generator, self).__init__()
        self.z_dim = z_dim # Latent sample size
        self.gf_dim = gf_dim # Generator sample size
        self.color_dim = color_dim # Color channels
        self.attr_range = attr_range # Range of value of attr - attr class
        self.n_attr = n_attr # 27 attrs 
        self.attr_size = attr_size # size to use for each attr

        # List of Conditions/Attributes
        # Attribute will have size: attr_range x attr_size --> e.g:
        # If attribute has values (0,1,2,3) -> attr_range = 4
        # --> attribute layer has size of 4 x attr_size
        for idx in range(0, self.n_attr): # 0 -> 27
            key = 'fc_' + str(idx)
            value = nn.Linear(self.attr_range, self.attr_size, bias=False)
            setattr(self, key, value)

        # input: 100 + n_classes + 27 * attribute_size, 
        # kernel_size = 8x4, stride = 1, padding = 0
        # ouput: 512 layers of (8 x 4) images - 512 x 8 x 4
        self.convT1 = nn.ConvTranspose2d(self.z_dim + self.attr_size * self.n_attr,
                                        self.gf_dim * 8, (8,4), 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.gf_dim * 8, momentum=0.3)

        # input: 512 x 8 x 4
        # kernel_size = 4x4, stride = 2, padding = 1
        # ouput: 256 x 16 x 8
        self.convT2 = nn.ConvTranspose2d(self.gf_dim * 8, self.gf_dim * 4,
                                        4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.gf_dim * 4, momentum=0.3)

        # input: 256 x 16 x 8
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 128 x 32 x 16
        self.convT3 = nn.ConvTranspose2d(self.gf_dim * 4, self.gf_dim * 2,
                                        4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.gf_dim * 2, momentum=0.3)

        # input: 128 x 32 x 16
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 64 x 64 x 32
        self.convT4 = nn.ConvTranspose2d(self.gf_dim * 2, self.gf_dim,
                                        4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.gf_dim, momentum=0.3)

        # input: 64 x 64 x 32
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 3 x 128 x 64
        self.convT5 = nn.ConvTranspose2d(self.gf_dim, self.color_dim,
                                        4, 2, 1, bias=False)

    # overload forward()
    def forward(self, x, y_n):
        cat_lst = [x, ]

        for i, y_i in enumerate(y_n):
            attr = self.__dict__['_modules']['fc_' + str(i)]
            val = F.leaky_relu(attr(y_i.squeeze()), 0.2, True)
            cat_lst.append(val.view(-1, self.attr_size, 1, 1))

        x = torch.cat(tuple(cat_lst), dim=1)

        x = F.leaky_relu(self.bn1(self.convT1(x)), 0.2, True)
        x = F.leaky_relu(self.bn2(self.convT2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.convT3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.convT4(x)), 0.2, True)

        x = F.tanh(self.convT5(x))

        return x


# Downsampling
class Discriminator(nn.Module):
    def __init__(self, df_dim=64, color_dim=3, 
                attr_range=4, n_attr=27, attr_size=4):

        super(Discriminator, self).__init__()
        self.df_dim = df_dim
        self.color_dim = color_dim
        self.attr_range = attr_range
        self.n_attr = n_attr # 27 attrs 
        self.attr_size = attr_size
        
        # input: 3 x 128 x 64
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 64 x 64 x 32
        self.conv1 = nn.Conv2d(self.color_dim, self.df_dim, 
                                4, 2, 1, bias=False)
        
        # input: 64 x 64 x 32
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 128 x 32 x 16
        self.conv2 = nn.Conv2d(self.df_dim, self.df_dim * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.df_dim * 2, momentum=0.3)
        
        # input: 128 x 32 x 16
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 256 x 16 x 8
        self.conv3 = nn.Conv2d(self.df_dim * 2, self.df_dim * 4, 
                                            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.df_dim * 4, momentum=0.3)

        # input: 256 x 16 x 8
        # kernel_size = 4x4, stride = 2, padding = 1
        # output: 512 x 8 x 4
        self.conv4 = nn.Conv2d(self.df_dim * 4, self.df_dim * 8,
                                            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.df_dim * 8, momentum=0.3)

        # input: 512 + 27 * attribute_size
        # kernel_size = 1, stride = 1, padding = 0
        # output: 512
        self.conv5 = nn.Conv2d(self.df_dim * 8 + self.attr_size * self.n_attr, 
                                            self.df_dim * 8, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(self.df_dim * 8, momentum=0.3)

        # input: 512 x 8 x 4
        # kernel_size = 8x4, stride = 1, padding = 0
        # output: 512 x 8 x 4
        self.conv6 = nn.Conv2d(self.df_dim * 8, 1, (8,4), 1, 0, bias=False)
        
        # List of Conditions/Attributes
        for idx in range(0, self.n_attr):
            key = 'fc_' + str(idx)
            value = nn.Linear(self.attr_range, self.attr_size, bias=False)
            setattr(self, key, value)

    #overload forward()
    def forward(self, x, y_n):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        cat_lst = []
        for i, y_i in enumerate(y_n):
            attr = self.__dict__['_modules']['fc_' + str(i)]
            val = F.leaky_relu(attr(y_i.squeeze()), 0.2, True)
            cat_lst.append(val)

        y = torch.cat(tuple(cat_lst), dim=1)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y_fill = y.repeat(1, 1, 8, 4)
        x = torch.cat((x, y_fill), dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        x = F.sigmoid(self.conv6(x))
        return x


class DCGAN(object):
    def __init__(self, z_dim=100, gf_dim=64, df_dim=64, 
                color_dim=3, beta1=0.5, 
                attr_range=4, n_attr=27, attr_size=4,
                device=torch.device("cuda:0")):
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.color_dim = df_dim
        self.beta1 = beta1
        self.attr_range = attr_range
        self.n_attr = n_attr # 27 attrs
        self.attr_size = attr_size
        self.device = device
        
        # Generator
        self.generator = Generator(attr_range=self.attr_range, 
                                    n_attr=self.n_attr,
                                    attr_size=self.attr_size).to(self.device)
        self.generator.apply(self.weights_init)
        
        # Discriminator
        self.discriminator = Discriminator(attr_range=self.attr_range, 
                                            n_attr=self.n_attr,
                                            attr_size=self.attr_size).to(self.device)
        self.discriminator.apply(self.weights_init)

        # Optimizer for Generator and Discriminator
        self.optimizerG = optim.Adam(self.generator.parameters(),
                                        lr=0.0008, betas=(self.beta1, 0.999))
        # ---> Discriminator learns slower
        self.optimizerD = optim.Adam(self.discriminator.parameters(),
                                        lr=0.0001, betas=(self.beta1, 0.999))
    
    def weights_init(self, w):
        # classname = w.__class__.__name__
        if (type(w) == nn.ConvTranspose2d or type(w) == nn.Conv2d):
            nn.init.normal_(w.weight.data, 0.0, 0.02)
        elif (type(w) == nn.BatchNorm2d):
            nn.init.normal_(w.weight.data, 1.0, 0.02)
            nn.init.constant_(w.bias.data, 0)
        elif (type(w) == nn.Linear):
            nn.init.normal_(w.weight.data, 0.0, 0.02)

    # Print out Summary
    def __str__(self):
        return str(self.generator) + '\n' + str(self.discriminator)
    
# dcGAN = DCGAN()
# print(dcGAN.__str__())
