from utils import pytorch_ssim

import torch
import torch.nn.functional as F

class CustomLoss():
    def __init__(self, recon_loss_name):
        
        if recon_loss_name == "L1":
            self.recon_loss_func = lambda x, y: torch.mean(torch.sum(torch.abs(x - y), dim=(1,2,3)), dim=0)
            
        elif recon_loss_name == "MSE":
            self.recon_loss_func = lambda x, y: torch.mean(torch.sum(torch.abs(x - y)**2, dim=(1,2,3)), dim=0)
            
        elif recon_loss_name == "BCE":
            raise NotImplementedError
            
        elif recon_loss_name == "SSIM":
            ssim = pytorch_ssim.SSIM(window_size = 3).cuda()
            self.recon_loss_func = lambda x, y: torch.mean(torch.sum(1 - ssim(x, y), dim=(1,2,3)), dim=0)
            
        elif recon_loss_name == "custom":
            raise NotImplementedError
    
    def __call__(self, x, y, mu, log_var):
        recons_loss = self.recon_loss_func(x, y)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        loss = recons_loss + kld_loss
        return loss, recons_loss, -kld_loss