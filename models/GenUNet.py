import torch
from torch import nn
import torch.nn.functional as F

#Single convolutional block
class SingleConv(nn.Module):
    """Creating a single convolutional block for the Generative UNet.
    Input:
        - in_channels: (int) number of channels in the input.
        - out_channels : (int) number of channels in the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
          )

    def forward(self, x):
        return self.single_conv(x)
    
#UNet Down layer
class UNetDown(nn.Module):
    """Creating a descending block for the Generative UNet.
    Input:
        - in_channels: (int) number of channels in the input.
        - out_size : (int) number of channels in the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UNetDown, self).__init__()
        self.down = SingleConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.down(x)

#UNet Up layer
class UNetUp(nn.Module):
    """Create an ascending block for the Generative UNet.
    Input:
        - in_size: (int) number of channels in the input.
        - out_size : (int) number of channels in the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super(UNetUp, self).__init__()
        self.up = SingleConv(in_channels, out_channels, kernel_size)
        
    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.up(x)
        return x

#Final layer
class FinalLayer(nn.Module):
    """Creating a final block of the Generative UNet.
    Input:
        - in_channels: (int) number of channels in the input.
        - out_size : (int) number of channels in the output.
    """
    def __init__(self, in_channels, out_channels):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

#Creating a Generator UNer
class GeneratorUNet(nn.Module):
    """Creating Generative UNet model.
    Input:
        - in_channels: (int) number of channels in the input.
        - out_channels : (int) number of channels in the output.
    """
    def __init__(self, in_channels=12, out_channels=1):
        super(GeneratorUNet, self).__init__()
        
        #down layers
        self.down1 = UNetDown(in_channels,64)
        self.down2 = UNetDown(64,128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512)
        self.down5 = UNetDown(512,512)

        #up layers
        self.up1 = UNetUp(512,512)
        self.up2 = UNetUp(1024,256)
        self.up3 = UNetUp(512,128)
        self.up4 = UNetUp(256,64)

        #final layer
        self.final_layer = FinalLayer(128, out_channels)


    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1,d4)
        u3 = self.up3(u2,d3)
        u4 = self.up4(u3,d2)

        return self.final_layer(u4,d1)
