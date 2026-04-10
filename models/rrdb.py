import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary    

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.out_conv = nn.Conv2d(nf + 3 * gc, nf, 1, bias=bias)  
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        out = self.out_conv(torch.cat((x, x1, x2, x3), 1))  
        return out * self.gamma + x


class GroupResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16):
        super(GroupResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(nf * 3, nf, 1, bias=True) 

    def forward(self, x):
        x1 = self.RDB1(x)
        x2 = self.RDB2(x1)
        x3 = self.RDB3(x2)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.out_conv(out)
        return out * self.gamma + x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        # 模拟全连接
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  
        return torch.sigmoid(out)  
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        out = torch.cat([avg_out, max_out], dim=1)  
        out = self.conv(out) 
        return torch.sigmoid(out)  

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=5):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  
        x = x * self.spatial_attention(x)  
        return x

class GroupResidualDenseNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32, gc=16, nb=3):
        super(GroupResidualDenseNet, self).__init__()
        self.prev_conv = nn.Conv2d(in_nc, nf//2, 3, 1, 1, bias=True)
        self.down_conv = nn.Sequential(
            nn.Conv2d(nf//2, nf, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.GRDB_trunk = nn.Sequential(*[GroupResidualDenseBlock(nf, gc) for _ in range(nb)])
        self.up_conv = nn.ConvTranspose2d(nf, nf//2, 4, stride=2, padding=1)
        self.res_conv = nn.Conv2d(in_nc, nf//2, 1)
        self.cbam = CBAM(nf, reduction=4)
        self.out_conv = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
    def forward(self, x):
        out = self.down_conv(self.prev_conv(x))
        out = self.GRDB_trunk(out)
        out = self.up_conv(out)
        out = self.cbam(torch.cat([out, self.res_conv(x)], dim=1))
        out = self.out_conv(out)
        return out + x
        
        
if __name__ == '__main__':
    model = GroupResidualDenseNet(3, 3)
    summary(model, (3, 128, 128), device='cpu')