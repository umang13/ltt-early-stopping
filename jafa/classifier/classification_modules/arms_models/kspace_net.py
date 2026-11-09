import torch
from torch._C import _ImperativeEngine
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import models

from .fft_conv import FFTConv2d
from classifier.utils import transforms
from classifier.utils.utils import string_to_list


class kSpaceNet(nn.Module):
    """
    KSpace net from paper:

    The workflow of this class is for an input (masked)K-space data, we perform fft conv based on:
    h = IFT(x dot Z'_F), where x is our input, dot is the element-wise dot product, Z'_F = F(Z').
    Here Z is our kernel, Z' is zero padded kernel.

    Then we convert complex h to real, center crop it to image size and then follows the traditional classification
    model.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: (Namespace) object holding attributes of yaml file

        Instance variable:
            num_classes: (int)      Number of class. For MRI data module, this is 2, whether you have disease or not.
            num_labels: (int)       For MRI data module, this is the number of diseases this model need to classify.
            label_names: [string]   List of disease name that this model need to classify.
                                    Note: len(label_names)=num_labels
            conv_kspace: (nn.Module)    Perform h = IFT(x dot Z'_F). Input: x, (masked)k-space data, Output: h
            backbone:               Traditional feature extractor. ["resnet50", "alexnet", "vit_b_16"]
        """
        super(kSpaceNet, self).__init__()
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.label_names = string_to_list(config.label_names)
        self.config = config

        if (
            self.config.dataset == "cifar10"
            or self.config.dataset == "imagenet"
            or self.config.dataset == "imagenet_large"
        ):
            in_channels = 3
            out_channels = 3
        else:
            in_channels = 1
            out_channels = 1

        self.conv_kspace = FFTConv2d(
            in_channels, out_channels, kernel_size=5, stride=1, bias=False
        )
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=self.config.image_shape
        )
        self.dropout = nn.Dropout(p=self.config.dropout)

        if self.config.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False
        print(f"Pretrained: {pretrained}, using imagenet weights")
        weights = "IMAGENET1K_V1" if pretrained else None

        if self.config.model_type == "resnet50":
            resnet50 = models.resnet50(weights=weights)
            resnet50.maxpool = nn.Identity()
            resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            num_ftrs = 2048
        elif self.config.model_type == "resnet18":
            resnet18 = models.resnet18(weights=weights)
            resnet18.maxpool = nn.Identity()
            resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512
        elif self.config.model_type == "alexnet":
            self.backbone = models.alexnet(weights=weights).features
            num_ftrs = 256
        elif self.config.model_type == "vit_b_16":
            vit = models.vit_b_16(weights=weights)
            self.backbone = nn.Sequential(*list(vit.children())[:-1])
            num_ftrs = 768
        else:
            raise NotImplementedError(
                f"Model type {self.config.model_type} not implemented"
            )

        self.dropout = nn.Dropout2d(p=self.config.dropout)

        self.label_layers = (
            nn.ModuleList()
        )  ## List of linear layers for multiple output heads
        for _ in range(0, self.config.num_labels):
            self.label_layers.append(nn.Linear(num_ftrs, self.config.num_classes))

    def forward(self, kspace):
        """

        :param kspace: (complex tensor) of size(batch, channel, height, width), channel=1 for MRI data
        :return: (dict) {disease name: prob for each class}
        """
        # assert len(kspace.shape) == 3, "k-space data must be of shape (batch_size, height, width)"

        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))

        if len(kspace.shape) == 3:
            kspace = kspace.unsqueeze(1)
        # print("kspace.shape in kspace net forward", kspace.shape)
        out_complex = self.conv_kspace(kspace)

        # if self.config.dataset in ["cifar10", "imagenet", "imagenet_large"]:
        #     out_mag = out_complex.real
        # elif self.config.dataset in ["knee", "brain", "prostate_t2"]:
        #     out_mag = out_complex.abs()
        # else:
        #     raise NotImplementedError(
        #         f"Dataset {self.config.dataset} not implemented for kSpaceNet"
        #     )
        
        out_mag = out_complex.abs()
        
        out_mag = transforms.center_crop(out_mag, self.config.image_shape)
        # out_mag = self.layernorm(out_mag)

        if out_mag.shape[1] == 1:
            out = out_mag.repeat(1, 3, 1, 1)
        else:
            out = out_mag
        out = self.backbone(out)
        # print("out.shape at predropout kspaceNet", out.shape)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)

        out_dict = {}

        for i in range(0, len(self.label_layers)):
            out_dict[self.label_names[i]] = self.label_layers[i](out)

        return out_dict
