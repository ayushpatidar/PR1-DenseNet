import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torchviz import make_dot


# DenseBlocks
class DenseBlock(nn.Module):
    def __init__(self, growth_rate, in_channels, out_channels, bottleleck_flag=True):
        
        super(DenseBlock, self).__init__()
        
        if bottleleck_flag == False:
            self.dense_network = nn.Sequential(nn.BatchNorm2d(in_channels),
                                                nn.ReLU(),
                                                nn.Conv2d(in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = 3,
                                                padding = 1)
                                                )
        else:
            
            self.dense_network = nn.Sequential(nn.BatchNorm2d(in_channels),
                                                nn.ReLU(),
                                                nn.Conv2d(in_channels = in_channels,
                                                out_channels = growth_rate*4,
                                                kernel_size = 1,
                                                padding = 0),
                                               
                                                nn.BatchNorm2d(growth_rate*4),
                                                nn.ReLU(),
                                                nn.Conv2d(in_channels = growth_rate*4,
                                                out_channels = out_channels,
                                                kernel_size = 3,
                                                padding = 1)
                                                )
    
        
    def forward(self, input):
        out = self.dense_network(input)
        return torch.cat([out, input], dim=1)
        
# TransitionBlocks 
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, stride):
        super(TransitionBlock, self).__init__()
        
        self.transition_network = nn.Sequential(
                                        nn.BatchNorm2d(in_channels),
                                        nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = 1,
                                        padding = padding,
                                        stride = stride),
                                        nn.AvgPool2d(kernel_size=2, stride=stride)
                                        )
        
    def forward(self, input):
        out = self.transition_network(input)
        return out
    
    
#ClassficationBlock
class ClassficationBlock(nn.Module):
    
    def __init__(self, in_channels, nclasses):
        super(ClassficationBlock, self).__init__()
        self.classification_network = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
                                                    nn.Flatten(),
                                                    nn.Linear(in_features = in_channels,
                                                            out_features = nclasses))
        
    def forward(self, input):
        out = self.classification_network(input)
        return out
        

#Main class
class Densenet(nn.Module):
    
    def __init__(self, growth_rate, input_shape, nclasses, compression = 0.5, bottleleck_flag=True,variant= 'Densenet-121'):
        super(Densenet, self).__init__()
    
        # K in the paper 
        self.growth_rate = growth_rate
        self.input_shape = input_shape
        self.selected_variant = variant
        self.compression = compression
        self.current_in_channels = 3
        self.current_out_channels = 3
        self.dense_concat_tensor = torch.Tensor()
        self.nclasses = nclasses
        self.variant_config = {
            "Densenet-121": [6,12,24,16],
            "Densenet-169": [6,12,32,32],
            "Densenet-201": [6,12,48,32],
            "Densenet-161": [6,12,36,24]
        }
        self.bottleleck_flag = bottleleck_flag
        
        
        self.current_out_channels = self.growth_rate*2
        self.initial_layer = nn.Sequential(nn.Conv2d(in_channels=self.current_in_channels,
                                                      out_channels=self.current_out_channels,
                                                      kernel_size = 7,
                                                      padding=1,
                                                      stride = 2),
                                           nn.MaxPool2d(kernel_size=3,
                                                       stride=2)                
                                          )
        
        
        self.current_in_channels = self.current_out_channels
        layers = list()
        
        for ind, blocks in enumerate(self.variant_config[self.selected_variant]):
            
            # Getting Dense Blocks
            for iter in range(blocks):
                # Dense layer
                layers.append(DenseBlock(self.growth_rate, self.current_in_channels, self.growth_rate, self.bottleleck_flag))
                self.current_in_channels   = self.current_in_channels + self.growth_rate
                
            # Getting Transition Blocks
            if ind!=3:
                # Transition layer
                layers.append(TransitionBlock(self.current_in_channels, int(self.compression*self.current_in_channels), 1, 1))
                self.current_in_channels   = int(self.compression*self.current_in_channels)
                
            else:
                #Classification layer
                layers.append(ClassficationBlock(self.current_in_channels, self.nclasses))
                
            
        self.network = nn.Sequential(*layers)
                        
                    

    def forward(self, input):
        
        # Have kept the Convoultion operation & Max pool hardcoded as it's constant for different variation of densenet
        out = self.initial_layer(input)
        out = self.network(out)
                
        return out