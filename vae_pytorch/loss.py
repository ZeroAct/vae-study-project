from utils import pytorch_ssim

ssim = pytorch_ssim.SSIM(window_size = 7).cuda()

def vae_loss(x, y):
    return 1 - ssim(x, y)