import torch
import torch.nn as nn
import torch.nn.functional as F

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False) 
        w = (w - m) / torch.sqrt(v + 1e-5) 
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock,self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)
        
        self.fuse_conv = nn.Sequential( #+ input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )
    
    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output

class L2HNet(nn.Module):
    def __init__(self, 
                 width, # width=64 for light mode; width=128 for normal mode
                 image_band=4, # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 output_chs=128, 
                 length=5, 
                 ratios=[1, 0.5, 0.25], 
                 bn_momentum=0.1):
        super(L2HNet,self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)
        
        self.out_conv1 = nn.Sequential(
            StdConv2d(self.width * length, output_chs * length, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, output_chs*5, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv2 = nn.Sequential(
            StdConv2d(output_chs * length, 1024, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv3 = nn.Sequential(
            StdConv2d(1024, 1024, kernel_size=5, stride=4, bias=False, padding=1),
            nn.GroupNorm(32,1024, eps=1e-6),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.startconv(x)
        output_d1 = []
        for rpblk in self.rpblocks:
            x = rpblk(x)
            output_d1.append(x)
        output_d1 = self.out_conv1(torch.cat(output_d1, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        features = [output_d1, output_d2, output_d3, x]
        return output_d3, features[::-1]
