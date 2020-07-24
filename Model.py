#!/usr/bin/env python3

from __future__ import print_function
import torch
import torch.nn as nn

num_classes = 3
n_layers_enc = 32
n_layers_ctx = 128
n_input = 6
prob_drop = 0.25
#experiment with group param in conv2d


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet,self).__init__()
        self.math=nn.Sequential(
            nn.Conv2d(n_layers_enc,n_layers_ctx,3,stride=1,padding=1,dilation=1),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #added layer
            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(1,2),dilation=(1,2)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),

            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(2,4),dilation=(2,4)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #nn.Dropout2d(prob_drop,inplace=True),
            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(4,8),dilation=(4,8)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #nn.Dropout2d(prob_drop,inplace=True),
            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(8,16),dilation=(8,16)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #nn.Dropout2d(prob_drop,inplace=True),
            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(16,32),dilation=(16,32)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #nn.Dropout2d(prob_drop,inplace=True),
            nn.Conv2d(n_layers_ctx,n_layers_ctx,3,stride=1,padding=(32,64),dilation=(32,64)),
            nn.ELU(),
            nn.AlphaDropout(prob_drop,inplace=True),
            #nn.Dropout2d(prob_drop,inplace=True),
            nn.Conv2d(n_layers_ctx,n_layers_enc,1),
            nn.ELU()
        )
    def forward(self,x):
        return self.math(x)

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet,self).__init__()
        self.math=nn.Sequential(
            nn.Conv2d(n_input,n_layers_enc,3,1,padding=1),
            nn.ELU(),
            nn.Conv2d(n_layers_enc,n_layers_enc,3,1,padding=1),
            nn.ELU()
        )
    def forward(self,x):
        return self.math(x)

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet,self).__init__()
        self.math=nn.Sequential(
            nn.Conv2d(n_layers_enc,n_layers_enc,3,stride=1,padding=1,dilation=1),
            nn.ELU(),
            nn.Conv2d(n_layers_enc,num_classes,3,stride=1,padding=1,dilation=1)
        )
    def forward(self,x):
        return self.math(x)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.Encoder=EncoderNet()
        self.Maxpool=nn.MaxPool2d(2,stride=2,return_indices=True)
        self.Context=ContextNet()
        self.MaxUnpool=nn.MaxUnpool2d(2,stride=2)
        self.Decoder=DecoderNet()
        self.Softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.Encoder(x)
        x,indices=self.Maxpool(x)
        x=self.Context(x)
        x=self.MaxUnpool(x,indices)
        x=self.Decoder(x)
        x=self.Softmax(x)
    #    self.features=getModel()
        return x
