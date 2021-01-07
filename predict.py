import os, argparse, time, cv2

import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.config import read_config
from utils.functions import tensor_to_mat, postprocess_image

from vae_pytorch.vae import Vae
from vae_pytorch.dataset import CustomDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", default="./configs/vae_test.ini", required=False, help=".ini file path")
    parser.add_argument("-d", "--data_path", default="datasets/vae_test", type=str, help="test data path")
    parser.add_argument("-n", "--num_workers", default=2, type=int, help="num_workers for DataLoader")
    
    args = parser.parse_args()
    
    return args
    

if __name__ == "__main__":
    
    args = get_args()
    
    model_path      = "./logs/" + args.config_file.split('/')[-1].split('.')[0]
    result_path     = "./results/" + args.config_file.split('/')[-1].split('.')[0]
    
    cfg = read_config(args.config_file)
    
    input_size     = cfg["model"]["input_size"]
    batch_size     = cfg["hyperparameters"]["batch_size"]
    
    dataloader_params = {'batch_size': batch_size,
                     'shuffle': True,
                     'drop_last': True,
                     'num_workers': args.num_workers}
    
    test_data = CustomDataset(args.data_path, input_size)
    test_gen = DataLoader(test_data, **dataloader_params)
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}...")
    
    vae = Vae(**cfg["model"]).to(device)
    _ = vae.load_weights(model_path)
    
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    
    idx = 0
    vae.eval()
    pgbar = tqdm.tqdm(test_gen, total=len(test_gen))
    for data in pgbar:
        data = data.to(device)
        recon, mu, log_var = vae.forward(data)
        data = data.cpu(); recon = recon.cpu(); mu = mu.cpu().detach().numpy(); log_var = log_var.cpu().detach().numpy()
        
        data = postprocess_image(tensor_to_mat(data))
        recon = postprocess_image(tensor_to_mat(recon))
        
        for x, y, mu, log_var in zip(data, recon, mu, log_var):
            cv2.imwrite(os.path.join(result_path, f"{idx}.png"), np.hstack([x, y])[:,:,::-1])
            with open(os.path.join(result_path, f"{idx}.txt"), 'w') as f:
                f.write("mu\n")
                f.write(",".join(list(map(str, list(mu)))))
                f.write("\n")
                f.write("log_var\n")
                f.write(",".join(list(map(str, list(log_var)))))
            idx += 1