import os
import torch
import glob
import cv2
import numpy as np
import torchvision.datasets as ds

from vae import Vae
from utils import preprocess_image, postprocess_image, mat_to_tensor, tensor_to_mat

import pytorch_ssim


input_size    = 256
in_channel    = 3
latent_dim    = 100
hidden_dims   = [64, 64, 128, 128, 256, 256]
device        = "cuda" if torch.cuda.is_available() else "cpu"

model_path  = "models/test_batchnorm"
test_img_dir = "../ExtractFrames/result2"

vae = Vae(input_size=input_size,
          in_channel=in_channel,
          latent_dim=latent_dim,
          hidden_dims=hidden_dims).to(device).eval()

if not os.path.isdir(model_path):
    os.mkdir(model_path)
else:
    vae.load_weights(model_path)
    print("loaded")
    
ssim = pytorch_ssim.SSIM(window_size = 7).cuda()
recon_loss = lambda x, y: 1 - ssim(x, y)

if test_img_dir == "CIFAR10":
    datas = ds.CIFAR10("datasets/CIFAR10", train=False, download=True).data
    
    for ori in datas:
        data = ori.copy()
        data = preprocess_image(data, input_size)
        data = mat_to_tensor(data)
        
        recon = vae.forward(data.to(device))
        print(f"SSIM loss: {recon_loss(data.to(device), recon)}")
        
        recon = tensor_to_mat(recon.cpu())[0]
        recon = postprocess_image(recon)
        
        ori = cv2.resize(ori, (input_size*2, input_size*2))
        recon = cv2.resize(recon, (input_size*2, input_size*2))
        
        cv2.imshow("result", np.hstack([ori, recon]))
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()

else:
    for test_img_path in glob.glob(test_img_dir + "/*.png"):
        ori = cv2.imread(test_img_path)
        
        data = ori.copy()
        data = preprocess_image(data, input_size)
        data = mat_to_tensor(data)
        
        recon = vae.forward(data.to(device))
        print(f"SSIM loss: {recon_loss(data.to(device), recon)}")
        
        recon = tensor_to_mat(recon.cpu())[0]
        recon = postprocess_image(recon)
        
        ori = cv2.resize(ori, (input_size*2, input_size*2))
        recon = cv2.resize(recon, (input_size*2, input_size*2))
        
        cv2.imshow("result", np.hstack([ori, recon]))
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()