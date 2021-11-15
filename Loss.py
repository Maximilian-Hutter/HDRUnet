import torch

def Tanh_Loss(gen_hdr, hdr):
    out = torch.abs(torch.tanh(gen_hdr) - torch.tanh(hdr))