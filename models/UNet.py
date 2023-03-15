import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """This class builds a double convolutional block (i.e. two single convolutional block).
    Each convolution layer is made up of a convolutional layer, a BatchNormalisation layer and a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels):
      super(DoubleConv,self).__init__()
      self.double_conv = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )

    def forward(self, x):
      return self.double_conv(x)


class DownBlock(nn.Module):
    """
    This class builds Downscaling blocks. 
    Each downscaling block is made up of a MaxPool layer and a DoubleConv block.
    """

    def __init__(self, in_channels, out_channels):
      super(DownBlock,seelf).__init__()
      self.maxpool_conv = nn.Sequential(
          nn.MaxPool2d(kernel_size=2),
          DoubleConv(in_channels, out_channels)
      )

    def forward(self, x):
      return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """This class build Upscaling blocks.
    Each downscaling block is made up of a ConvTranspose layer and a DoubleConv block.
    This layer has the possibility to include skip connection.
    """

    def __init__(self, in_channels, out_channels):
      super(UpBlock,self).__init__()

      self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
      x1 = self.up(x1)
      #adding skip connection
      if x2 is not None:
        x = torch.cat([x2, x1], dim=1)
      x = self.conv(x)

      return x


class FinalBlock(nn.Module):
  """
  This class builds a Final Convolutional Block.
  This block is made up of a single convolutional layer.
  """
    def __init__(self, in_channels, out_channels):
      super(FinalBlock, self).__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
      return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #down blocks
        self.conv = DoubleConv(in_channels, 64)

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        
        #up block
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        #final block
        self.out_conv = FinalBlock(64, out_channels)
      
   def forward(self, x):
    
    #down blocks
    d1 = self.conv(x)
    d2 = self.down1(d1)
    d3 = self.down2(d2)
    d4 = self.down3(d3)
    d5 = self.down4(d4)
    
    #up blocks
    u1 = self.up1(d5, d4)
    u2 = self.up2(u1, d3)
    u3 = self.up3(u2, d2)
    u4 = self.up4(u3, d1)
    
    #final layer
    out = self.out_conv(u4)
    
    return out
