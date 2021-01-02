from configparser import ConfigParser

import json

def create_default_config():
    cfg = ConfigParser()
    cfg["model"] = {
            "input_size": '[32, 32]',
            "in_channel": '3',
            "latent_dim": '10',
            "hidden_dims": '[32, 32, 32]',
        }
    
    cfg["hyperparameters"] = {
            "epochs": '5',
            "batch_size": '2',
            "optimizer": "adam",
            "learning_rate": '1e-2',
        }
    
    cfg["data"] = {
            "train_img_path": 'None',
            "val_img_path": 'None'
        }
    
    with open("./configs/default.ini", "w") as f:
        cfg.write(f)

def read_config(cfg_path):
    result = {}
    
    cfg = ConfigParser()
    cfg.read(cfg_path)
    
    def get_(s, o):
        value = cfg.get(s, o)
        try:
            value = json.loads(value)
        except:
            if value == "None": value = None
        return value
    
    cfg.sections()
    result["model"] = {
        "input_size":     get_("model", "input_size"),
        "in_channel":     get_("model", "in_channel"),
        "latent_dim":     get_("model", "latent_dim"),
        "hidden_dims":    get_("model", "hidden_dims")
    }
    
    result["hyperparameters"] = {
        "epochs":         get_("hyperparameters", "epochs"),
        "batch_size":     get_("hyperparameters", "batch_size"),
        "optimizer":      get_("hyperparameters", "optimizer"),
        "learning_rate":  get_("hyperparameters", "learning_rate")
    }
    
    result["data"] = {
        "train_img_path": get_("data", "train_img_path"),
        "val_img_path":   get_("data", "val_img_path")
    }
    
    return result

if __name__ == "__main__":
    cfg_path = "./configs/default.ini"
    create_default_config()
    cfg = read_config(cfg_path)
