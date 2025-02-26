import torch
import torch.nn as nn
from enum import Enum

__all__ = ['lenet1d', 'LeNet1d']

# Define NormType Enum
NormType = Enum('NormType', 'Batch BatchZero')

# Define BatchNorm function
def BatchNorm(nf, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, zero=norm_type==NormType.BatchZero, **kwargs)

# Helper function to get normalization layer
def _get_norm(prefix, nf, zero=False, **kwargs):
    "Norm layer with `nf` features initialized depending on `norm_type`."
    bn = getattr(nn, f"{prefix}1d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

# Initialize weights
def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not(bn)
        conv_func = nn.Conv1d
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs), init)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

class LeNet1d(nn.Module):
    def __init__(self, input_channels, num_classes, bn=True, ps=False, hidden=False, 
                 c1=12, c2=32, c3=64, c4=128, k=5, s=2, E=128):
        super().__init__()
        
        self.bn = bn
        self.ps = ps
        self.hidden = hidden
        
        modules = []
        modules.append(self.conv_block(input_channels, c2, k, s, bn, ps))
        modules.append(self.conv_block(c2, c3, k, s, bn, ps))
        modules.append(self.conv_block(c3, c4, k, s, bn, ps, final=True))
        
        self.conv_layers = nn.Sequential(*modules)

        self.flatten = nn.Flatten()

        # Adjust input size of fc1 to match the output size of conv layers
        self.fc1 = nn.Linear(4736, E)
        self.fc2 = nn.Linear(E, num_classes)
        
        if hidden:
            self.hidden_layers = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(E, E),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.hidden_layers = None

    def conv_block(self, in_channels, out_channels, kernel_size, stride, bn, ps, final=False):
        layers = [init_default(nn.Conv1d(in_channels, out_channels, kernel_size, stride))]
        if bn:
            layers.append(BatchNorm(out_channels, norm_type=NormType.Batch))
        layers.append(nn.ReLU())
        if not final:
            layers.append(nn.MaxPool1d(2))
        if ps:
            layers.append(nn.Dropout(0.1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        #print(f"Shape after conv layers: {x.shape}")
        x = self.flatten(x)
        #print(f"Shape after flatten: {x.shape}")
        x = self.fc1(x)
        x = nn.ReLU()(x)
        if self.hidden_layers:
            x = self.hidden_layers(x)
        x = self.fc2(x)
        return x

    def freeze_backbone(self):
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

def lenet1d(input_channels, num_classes):
    return LeNet1d(input_channels=input_channels, num_classes=num_classes)
