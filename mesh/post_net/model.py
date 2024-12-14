import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

# Refer: https://github.com/twtygqyy/pytorch-vdsr/blob/master/vdsr.py
class VDSR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64+in_channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = torch.cat((out,residual), dim=1) # concat at channel dim
        out = self.output(out)
        out = out.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        return out

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_C8(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_C16(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_K3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=3 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=3, padding=3 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_K7(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=7, padding=7 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=7, padding=7 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_K9(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=9, padding=9 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 3, kernel_size=9, padding=9 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_L(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, padding=5 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN_S(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        orig = x[...,:3]
        x = x.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x = self.bn1(self.conv1(x))
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x

# Refer: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class Gated_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=5 // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv_g = nn.Conv2d(1, 1, kernel_size=5, padding=5 // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x[...,:3]
        orig_depth = x[...,3].unsqueeze(-1)
        x1 = orig.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
        x1 = self.relu(self.bn1(self.conv1(x1)))
        x2 = orig_depth.permute(0, 3, 1, 2)
        x2 = self.sigmoid(self.conv_g(x2))
        x = x1 * x2
        x = x.permute(0, 2, 3, 1) # N, C, H, W -> N, H, W, C
        x = orig + 0.1* x
        return x