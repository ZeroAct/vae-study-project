import os, argparse, time

import tqdm

import torch
from torch.utils.data import DataLoader

from utils.config import read_config
from vae_pytorch.vae import Vae
from vae_pytorch.dataset import CustomDataset
from vae_pytorch.loss import vae_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./configs/default.ini", required=False, help=".ini file path")
    parser.add_argument("-v", "--val_interval", default=1, type=int, help="val interval(epoch)")
    parser.add_argument("-s", "--save_interval", default=1, type=int, help="val interval(epoch)")
    parser.add_argument("-n", "--num_workers", default=2, type=int, help="num_workers for DataLoader")
    
    args = parser.parse_args()
    
    return args
    

if __name__ == "__main__":
    
    args = get_args()
    
    save_path      = "./logs/" + args.config_file.split('/')[-1].split('.')[0]
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
        val_data = CustomDataset(train_img_path, input_size)
        val_gen = DataLoader(val_data, **dataloader_params)
    
    steps_per_epoch = len(train_gen)
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}...")
    
    vae = Vae(**cfg["model"]).to(device)
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    else:
        initial_epoch = vae.load_weights(save_path)
    
    # Train Setting
    if optimizer == "adam":
        optim = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    loss_function = vae_loss
    
    # Train
    train_losses = [9999]
    val_losses = [9999]
    
    for epoch in range(initial_epoch, epochs):
        # training
        train_loss = []
        
        vae.train()
        pgbar = tqdm.tqdm(train_gen, total=len(train_gen))
        pgbar.set_description(f"Epoch {epoch+1}/{epochs}")
        for data in pgbar:
            optim.zero_grad()
            
            data = data.to(device)
            recon = vae.forward(data)
            
            loss = loss_function(data, recon)
            loss.backward()
            optim.step()
            
            train_loss.append(loss.item())
            
            pgbar.set_postfix_str(f"loss : {sum(train_loss[-10:]) / len(train_loss[-10:]):.6f}")
        
        train_losses.append(sum(train_loss) / len(train_loss))
        
        # validation
        if val_interval != 0 and (epoch + 1) % val_interval == 0:
            val_loss = []
            
            vae.eval()
            pgbar = tqdm.tqdm(val_gen, total=len(val_gen))
            pgbar.set_description("Validating...")
            for data in pgbar:
                data = data.to(device)
                recon = vae.forward(data)
                
                loss = loss_function(data, recon)
                
                val_loss.append(loss.item())
                
                pgbar.set_postfix_str(f"loss : {sum(val_loss[-10:]) / len(val_loss[-10:]):.6f}")
            
            val_losses.append(sum(val_loss) / len(val_loss))
        
        print(f"train_loss : {train_losses[-1]}, val_loss : {val_losses[-1]}")
        print()
        time.sleep(0.2)
        
        if (epoch + 1) % save_interval == 0:
            torch.save(vae.state_dict(), os.path.join(save_path, f"{epoch}_train_{train_losses[-1]}_val_{val_losses[-1]}.pth"))
        
        
        