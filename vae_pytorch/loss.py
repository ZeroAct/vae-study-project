from utils import pytorch_ssim

import torch


class CustomLoss():
    def __init__(self, recon_loss_name, kld_weight):
        
        if recon_loss_name == "L1":
            self.recon_loss_func = torch.nn.L1Loss()
            
        elif recon_loss_name == "MSE":
            self.recon_loss_func = torch.nn.MSELoss()
            
        elif recon_loss_name == "BCE":
            self.recon_loss_func = torch.nn.BCEWithLogitsLoss()
            
        elif recon_loss_name == "SSIM":
            ssim = pytorch_ssim.SSIM(window_size = 7).cuda()
            self.recon_loss_func = lambda x, y: 1 - ssim(x, y)
            
        elif recon_loss_name == "custom":
            raise NotImplementedError
        
        
        self.kld_weight = kld_weight
    
    def __call__(self, x, y, mu, log_var):
        recons_loss = self.recon_loss_func(x, y)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        loss = recons_loss + self.kld_weight * kld_loss
        return loss, recons_loss, kld_loss