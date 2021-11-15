import torch 
import torch.nn as nn
from Models import *

class HDRUnet(nn.Module):
    def __init__(self):
        super(HDRUnet, self).__init__()
               

        self.Conv1 = nn.Conv2d()
        self.SFT1 = SFT()
        self.Conv2 = nn.Conv2d()
        self.Down1 = Downsample()
        self.Res1 = ResBlock()
        self.Res2 = ResBlock()
        self.Down2 = Downsample()

        self.Resnet = Resnet()

        self.Conv3 = nn.Conv2d()
        self.Up1 = nn.Upsample()
        self.Res3 = ResBlock()
        self.Res4 = ResBlock()
        self.Up2 = nn.Upsample()
        self.SFT2 = SFT()
        self.Conv4 = nn.Conv2d()
        self.Conv5 = nn.Conv2d()
                
    def forward(self, x, BaseSFT, Down1SFT, Down2SFT, weightestim):      

        x = self.Conv1(x)
        x = self.SFT1(x, BaseSFT)
        x = self.Conv2(x)
        
        out1 = self.Down1(x) # in paper Conv with stride=2 is used for downsampling
        out2 = self.ResLayers1(out1, Down1SFT)
        out3 = self.Down2(out2)

        x = self.Resnet(out3, Down2SFT)

        x = torch.add(x ,out3)
        x = self.Conv3(x)
        x = self.Up1(x) # in paper pixel shuffle is used to upsample

        x = torch.add(x, out2)
        x = self.ResLayers2(x, Down1SFT)
        x = self.Up2(x) 

        x = torch.add(x, out1)
        out = self.SFT2(x, BaseSFT)

        out = self.Conv4(out)
        out = self.Conv5(out)
        out = torch.add(out, weightestim)

        return out