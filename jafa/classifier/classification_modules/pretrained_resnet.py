import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from classifier.utils.utils import string_to_list
from torchvision import transforms

class PreTrainedResNet(nn.Module):
    def __init__(self, config):
        super(PreTrainedResNet, self).__init__()
        
        # Handle imbalanced datasets: adjust num_classes for binary classification
        self.imbalanced_class = getattr(config, 'imbalanced_class', None)
        self.is_imbalanced = self.imbalanced_class is not None
        
        if self.is_imbalanced:
            print(f"PreTrainedResNet: Imbalanced dataset detected: {self.imbalanced_class}")
            print(f"PreTrainedResNet: Converting from {config.num_classes} classes to binary classification")
            self.num_classes = 2  # Binary classification
        else:
            self.num_classes = config.num_classes
            
        self.num_labels = config.num_labels
        self.label_names = string_to_list(config.label_names)        
        self.config = config

        if self.config.pretrained == "imagenet":
            pretrained = True
            #shaan
            print('++++++++++++++++++++++++++++++++')
            print('Using imagenet pretrained weights')
            print('++++++++++++++++++++++++++++++++')
        else:
            pretrained = False
            #shaan
            print('++++++++++++++++++++++++++++++++')
            print(' NOT Using imagenet pretrained weights')
            print('++++++++++++++++++++++++++++++++')
        print(f"Pretrained: {pretrained}, using imagenet weights")            
        weights = "IMAGENET1K_V1" if pretrained else None

        if self.config.model_type == 'resnet50':            
            resnet50 = models.resnet50(weights=weights)            
            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            num_ftrs = 2048
        elif self.config.model_type == "resnet18":
            resnet18 = models.resnet18(weights=weights)
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
            raise NotImplementedError(f"Model type {self.config.model_type} not implemented")
        

        # Shaan : commented out below since only need feature vectors
        # self.dropout = nn.Dropout2d(p=self.config.dropout_rate)
        # self.predict_sufficiency = getattr(self.config, 'predict_sufficiency', False)

    
        # self.label_layers = nn.ModuleList() ## List of linear layers for multiple output heads
        # for _ in range(0, self.config.num_labels):
        #     self.label_layers.append(nn.Linear(num_ftrs, self.num_classes))
        
        # print(f"PreTrainedResNet: Created {self.config.num_labels} output heads with {self.num_classes} classes each")
        
        # if self.predict_sufficiency:
        #     print("PreTrainedResNet: Enabling sufficiency prediction head")
        #     hidden_dim = 64
        #     self.sufficiency_head = nn.Sequential(
        #         nn.Linear(num_ftrs, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, 1)
        #     )
        

        # Shaan : added the below to freeze backbone 
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
    def forward(self, x):        

        if x.shape[1] == 1:
            batch_size = x.shape[0]
            out = x.repeat(1, 3, 1, 1).float()        
        elif self.config.dataset == "cifar10" or x.shape[1] == 3:
            batch_size = x.shape[0]
            out = x.float()   
        
        out = self.backbone(out)

        #Shaan added below code

        features = out
        # below should be (batch_size, 512, 1, 1). well it is (batch_size, 512)
        features = torch.flatten(features, 1)
        return features


        #Shaan commented out below code

        # out = self.dropout(out)

        # out = out.view(batch_size, -1)
        # sufficiency_scores = None
        # if self.predict_sufficiency:
        #     sufficiency_scores = self.sufficiency_head(out)
        
        # out_dict = {}
        
        # for i in range(0, len(self.label_layers)):
        #     out_dict[self.label_names[i]] = self.label_layers[i](out)            
        
        # if sufficiency_scores is not None:
        #     out_dict['sufficiency'] = sufficiency_scores
        
        # return out_dict 