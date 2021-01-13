import os, argparse, time, random

import tqdm

import torch
from torch.utils.data import DataLoader

from utils.config import read_config
from vae_pytorch.vae import Vae
from vae_pytorch.dataset import CustomDataset
from vae_pytorch.loss import CustomLoss

from utils.functions import tensor_to_mat, postprocess_image

import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./configs/cifar10.ini", required=False, help=".ini file path")
    parser.add_argument("-v", "--val_interval", default=1, type=int, help="val interval(epoch)")
    parser.add_argument("-s", "--save_interval", default=1, type=int, help="val interval(epoch)")
    parser.add_argument("-n", "--num_workers", default=8, type=int, help="num_workers for DataLoader")
    parser.add_argument("-o", "--show", default=True, type=bool, help="show reconstruct images during training")
    
    args = parser.parse_args()
    
    return args
    

if __name__ == "__main__":
    
    args = get_args()
    
    model_path     = "./logs/" + args.config_file.split('/')[-1].split('.')[0]
    val_interval   = args.val_interval
    save_interval  = args.save_interval
    num_workers    = args.num_workers
    
    cfg = read_config(args.config_file)
    
    input_size     = cfg["model"]["input_size"]
    
    epochs         = cfg["hyperparameters"]["epochs"]
    initial_epoch  = 0
    batch_size     = cfg["hyperparameters"]["batch_size"]
    optimizer      = cfg["hyperparameters"]["optimizer"]
    learning_rate  = cfg["hyperparameters"]["learning_rate"]
    loss_name      = cfg["hyperparameters"]["loss"]
    
    train_img_path = cfg["data"]["train_img_path"]
    val_img_path   = cfg["data"]["val_img_path"]
    
    if val_img_path is None: val_interval = 0
    
    # DataLoader
    assert train_img_path is not None
    
    dataloader_params = {'batch_size': batch_size,
                         'shuffle': True,
                         'drop_last': True,
                         'num_workers': num_workers}
    
    train_data = CustomDataset(train_img_path, input_size)
    train_gen = DataLoader(train_data, **dataloader_params)
    
    if val_img_path is not None:
        val_data = CustomDataset(val_img_path, input_size)
        val_gen = DataLoader(val_data, **dataloader_params)
    
    steps_per_epoch = len(train_gen)
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}...")
    
    vae = Vae(**cfg["model"]).to(device)
    
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(os.path.join(model_path, "weights"))
        os.mkdir(os.path.join(model_path, "show"))
    else:
        initial_epoch = vae.load_weights(os.path.join(model_path, "weights"))
    
    # Train Setting
    if optimizer == "adam":
        optim = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        
    loss_function = CustomLoss(loss_name)
    
    # Train
    train_losses = [9999]
    val_losses = [9999]
    
    for epoch in range(initial_epoch, epochs):
        # training
        train_loss = []
        recon_loss = []
        kld_loss   = []
        
        vae.train()
        pgbar = tqdm.tqdm(train_gen, total=len(train_gen))
        pgbar.set_description(f"Epoch {epoch}/{epochs}")
        for data in pgbar:
            optim.zero_grad()
            
            data = data.to(device)
            recon, mu, log_var = vae.forward(data)
            
            loss, recon_loss_, kld_loss_ = loss_function(data, recon, mu, log_var)
            loss.backward()
            optim.step()
            
            train_loss.append(loss.item())
            recon_loss.append(recon_loss_.item())
            kld_loss.append(kld_loss_.item())
            
            pgbar.set_postfix_str(f"loss : {sum(train_loss[-10:]) / len(train_loss[-10:]):.6f} recon_loss : {sum(recon_loss[-10:]) / len(recon_loss[-10:]):.6f} kld_loss : {sum(kld_loss[-10:]) / len(kld_loss[-10:]):.6f}")
        
        train_losses.append(sum(train_loss) / len(train_loss))
        
        # validation
        if val_interval != 0 and (epoch + 1) % val_interval == 0:
            val_loss = []
            
            # will saved in show directory
            shows = []
            
            vae.eval()
            pgbar = tqdm.tqdm(val_gen, total=len(val_gen))
            pgbar.set_description("Validating...")
            for data in pgbar:
                data = data.to(device)
                recon, mu, log_var = vae.forward(data)
                
                loss, recon_loss_, kld_loss_ = loss_function(data, recon, mu, log_var)
                
                val_loss.append(loss.item())
                
                pgbar.set_postfix_str(f"loss : {sum(val_loss[-10:]) / len(val_loss[-10:]):.6f}")
                
                if args.show and len(shows) < 5:
                    shows.append([data[:1], recon[:1]])
            
            val_losses.append(sum(val_loss) / len(val_loss))
            
            if args.show:
                fig, axs = plt.subplots(5, 1, figsize=(5, 8))
                fig.suptitle("original -> reconstruction")
                for i, (data, recon) in enumerate(shows):
                    data, recon = data.cpu(), recon.cpu()
                    data = postprocess_image(tensor_to_mat(data))[0][:,:,::-1]
                    recon = postprocess_image(tensor_to_mat(recon))[0][:,:,::-1]
                    
                    axs[i].imshow(np.hstack([data, recon]))
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                
                fig.savefig(os.path.join(model_path, "show", f"epoch_{epoch}.png"))
                
        
        print(f"train_loss : {train_losses[-1]}, val_loss : {val_losses[-1]}")
        print()
        time.sleep(0.2)
        
        if (epoch + 1) % save_interval == 0:
            torch.save(vae.state_dict(), os.path.join(model_path, f"weights/{epoch}_train_{train_losses[-1]}_val_{val_losses[-1]}.pth"))
        
        
        