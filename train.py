# import statements
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms, models
import numpy as np
import argparse
from tqdm import tqdm
import time

from tensorboardX import SummaryWriter
from Loss import Tanh_Loss

from utils import load_checkpoint, save_checkpoint, ensure_dir
from HDRModel import HDRUnet
from Models import Conditionalnet, WeightEstimationNet

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a network for image classification on cifar10.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Spcifies learing rate for optimizer. (default: 1e-3)')
 
    opt = parser.parse_args()

    # data loader

    
    # instantiate network (which has been imported from *networks.py*)
    HDRnet = HDRUnet()
    Conditnet = Conditionalnet()
    weightnet = WeightEstimationNet()
    
    # create losses (criterion in pytorch)
    criterion_hdr = Tanh_Loss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        HDRnet = HDRnet.cuda()
    
    # create optimizers
    optim = torch.optim.Adam(HDRnet.parameters(), lr=opt.lr)
    optim_Con = torch.optim.Adam(Conditnet.parameters(), lr=opt.lr)
    optim_weight = torch.optim.Adam(weightnet.parameters(), lr=opt.lr)
    
    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint) # custom method for loading last checkpoint
        HDRnet.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")
        
    
    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    
    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, opt.epochs):
        # set models to train mode
        HDRnet.train()
        
        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))
        start_time = time.time()
        
        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            ldr, hdr = data
            if use_cuda:
                ldr = ldr.cuda()
                hdr = hdr.cuda()
            
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time-time.time()
            
            # forward and backward pass
            BaseSFT, Down1SFT, Down2SFT = Conditionalnet(ldr)
            weightestim = WeightEstimationNet(ldr)
            gen_hdr = HDRnet(ldr,BaseSFT, Down1SFT, Down2SFT, weightestim)
            loss = criterion_hdr(gen_hdr, hdr)
            optim.zero_grad()
            optim_Con.zero_grad()
            optim_weight.zero_grad()
            loss.backward()
            optim.step()
            optim_Con.step()
            optim_weight.step()
            
            # udpate tensorboardX
            writer.add_scalar('train_loss', n_iter)
            
            # compute computation time and *compute_efficiency*
            process_time = start_time-time.time()-prepare_time
            compute_efficiency = process_time/(process_time+prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, ' 
                f'loss: {loss.item():.2f},  epoch: {epoch}/{opt.epochs}')
            start_time = time.time()
                
            # save checkpoint if needed
            cpkt = {
                'net': HDRnet.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
                'optim': optim.state_dict()
            }
            save_checkpoint(cpkt, 'model_checkpoint.ckpt')
