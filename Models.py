import torch.nn as nn
import torch

class SFT(nn.Module):    
    def __init__(self, ConMap):
        super(SFT, self).__init__()
               
        self.ConMap = ConMap
        
        self.Conv1 = nn.Conv2d()
        self.Conv2 = nn.Conv2d()

        self.Conv3 = nn.Conv2d()
        self.Conv4 = nn.Conv2d()
        
    def forward(self, x, ConMap):      

        Con1 = self.Conv1(ConMap)
        Con1 = self.Conv2(Con1)

        Con2 = self.Conv3(ConMap)
        Con2 = self.Conv4(Con2)

        x = torch.mul(x, Con1)
        out = torch.add(x, Con2)

        return out

class ResBlock(nn.Module): 
    def __init__(self, ConMap):
        super(ResBlock, self).__init__()

        self.SFT1 = SFT(ConMap)
        self.Conv1 = nn.Conv2d()
        self.SFT2 = SFT(ConMap)       
        self.Conv2 = nn.Conv2d()

    def forward(self, x):      

        out = self.SFT1(x)
        out = self.Conv1(x)
        out = self.SFT2(x)
        out = self.Conv2
        out = torch.add(out, x)

        return out

class WeightEstimationNet(nn.Module):  
    def __init__(self):
        super(WeightEstimationNet, self).__init__()
               
        self.Conv = ConvBlock
        
    def forward(self, x):      

        out = self.Conv(x)
        out = torch.mul(out, x)

        return out

class Resnet(nn.Module):   
    def __init__(self,n_resblock, ConMap):
        super(Resnet, self).__init__()
               
        self.ConMap = ConMap

        layers = []
        for _ in range(n_resblock):
            layers += ResBlock(ConMap)
        self.Res = nn.Sequential(*layers)

    def forward(self, x):      

        out = self.Res(x)

        return out

class BaseCondit(nn.Module):  
    def __init__(self):
        super(BaseCondit, self).__init__()

        self.Conv1 = nn.Conv2d()
        self.Conv2 = nn.Conv2d()
        self.Conv3 = nn.Conv2d() 

    def forward(self, x):      

        x = self.ConvBlock(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        out = self.Conv3(x)

        return out

class Down1Condit(nn.Module):   
    def __init__(self):
        super(Down1Condit, self).__init__()

        self.Down1 = Downsample()
        self.Conv1 = nn.Conv2d()
        self.Conv2 = nn.Conv2d()

    def forward(self, x):      

        x = self.ConvBlock(x)
        x = self.Down1(x)
        x = self.Conv2(x)
        out = self.Conv2(x)

        return out

class Down2Condit(nn.Module): 
    def __init__(self):
        super(Down2Condit, self).__init__()

        self.Down1 = Downsample()
        self.Down2 = Downsample()
        self.Conv1 = nn.Conv2d()   

    def forward(self, x):      

        x = self.ConvBlock(x)
        x = self.Down1(x)
        x = self.Down2(x)
        out = self.Conv1(x)

        return out

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.Conv1 = nn.Conv2d()
        self.Conv2 = nn.Conv2d()
        self.Conv3 = nn.Conv2d()

    def forward(self, x):

        x = self.Conv1(x)
        x = self.Conv2(x)
        out = self.Conv3(x)

        return out

class Downsample(nn.Module):
    def __init__(self, scale_factor):
        super(Downsample, self).__init__()

        scale_factor = 1/scale_factor

        self.Down = nn.Upsample(scale_factor)
    def forward(self, x):

        out = self.Down(x)

        return out

class Conditionalnet(nn.Module):
    def __init__(self):
        super(Conditionalnet, self).__init__()

        self.ConditConvBlock = ConvBlock()
        self.Condit1 = BaseCondit()
        self.Condit2 = Down1Condit()
        self.Condit3 = Down2Condit()

    def forward(self, x):
        x = self.ConditConvBlock(x)

        BaseSFT = self.Condit1(x)
        Down1SFT = self.Condit2(x)
        Down2SFT = self.Condit3(x)

        return BaseSFT, Down1SFT, Down2SFT