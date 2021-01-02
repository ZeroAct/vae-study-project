import os, torch, natsort, glob

from torch import nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Vae(nn.Module):
    def __init__(self, 
                 input_size:    list  = [64, 64],
                 in_channel:    int   = 3,
                 latent_dim:    int   = None,
                 hidden_dims:   list  = None):
        
        super(Vae, self).__init__()
        
        self.input_size    = input_size
        self.in_channel    = in_channel
        self.latent_dim    = latent_dim  if latent_dim  is not None else 10
        self.hidden_dims   = hidden_dims if hidden_dims is not None else [16, 32, 32]
        
        self.final_size    = [self.input_size[0] // (2 ** len(self.hidden_dims)),
                              self.input_size[1] // (2 ** len(self.hidden_dims))]
        self.hidden_dims   = [self.in_channel] + self.hidden_dims
        assert self.final_size[0] > 1 and self.final_size[1] > 1
        
        # Build Encoder
        self.encoder_blocks = []
        
        for i in range(len(self.hidden_dims) - 1):
            self.encoder_blocks.append(nn.Conv2d(self.hidden_dims[i],
                                                 self.hidden_dims[i+1],
                                                 3, 1, 1))
            self.encoder_blocks.append(nn.BatchNorm2d(self.hidden_dims[i+1]))
            self.encoder_blocks.append(nn.LeakyReLU()),
            self.encoder_blocks.append(nn.Conv2d(self.hidden_dims[i+1],
                                                 self.hidden_dims[i+1],
                                                 3, 2, 1))
            self.encoder_blocks.append(nn.BatchNorm2d(self.hidden_dims[i+1]))
            self.encoder_blocks.append(nn.LeakyReLU())
        
        self.encoder = nn.Sequential(*self.encoder_blocks)
        
        self.encoder_output = nn.Sequential(nn.Linear(self.final_size[0] * self.final_size[1] * self.hidden_dims[-1], self.latent_dim), 
                                            nn.BatchNorm1d(self.latent_dim), 
                                            nn.LeakyReLU())
        
        # Build Encoder
        self.decoder_input = nn.Sequential(nn.Linear(self.latent_dim, self.final_size[0] * self.final_size[1] * self.hidden_dims[-1]),
                                           nn.BatchNorm1d(self.final_size[0] * self.final_size[1] * self.hidden_dims[-1]), 
                                           nn.LeakyReLU(),
                                           Reshape((-1, self.hidden_dims[-1], self.final_size[0], self.final_size[1])))
        
        self.hidden_dims.reverse()
        
        self.decoder_blocks = []
        
        for i in range(len(self.hidden_dims) - 2):
            self.decoder_blocks.append(nn.ConvTranspose2d(self.hidden_dims[i],
                                                          self.hidden_dims[i+1],3,2,1,1))
            self.decoder_blocks.append(nn.BatchNorm2d(self.hidden_dims[i+1]))
            self.decoder_blocks.append(nn.LeakyReLU()),
            self.encoder_blocks.append(nn.Conv2d(self.hidden_dims[i+1],
                                                 self.hidden_dims[i+1],
                                                 3, 1, 1))
            self.encoder_blocks.append(nn.BatchNorm2d(self.hidden_dims[i+1]))
            self.encoder_blocks.append(nn.LeakyReLU()),
        
        self.decoder_blocks.append(nn.ConvTranspose2d(self.hidden_dims[-2],
                                                      self.hidden_dims[-1],3,2,1,1))
        self.decoder_blocks.append(nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], 3, 1, 1))
        self.decoder_blocks.append(nn.BatchNorm2d(self.hidden_dims[-1]))
        self.decoder_blocks.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*self.decoder_blocks)
        
        self.hidden_dims.reverse()
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        
        return x
        
    def encode(self, x):
        x = self.encoder(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_output(x)
        
        return x
    
    def decode(self, x):
        x = self.decoder_input(x)
        x = torch.reshape(x, (-1, self.hidden_dims[-1], self.final_size[0], self.final_size[1]))
        
        x = self.decoder(x)
        return x
    
    def debug(self, x):
        
        for i, b in enumerate(self.encoder_blocks):
            print(b)
            print(x.shape)
            try: x = b(x)
            except: print(self.encoder_blocks[i+1])
            print(x.shape)
            print()
        
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_output(x)
        
        x = self.decoder_input(x)
        x = torch.reshape(x, (-1, self.hidden_dims[-1], self.final_size[0], self.final_size[1]))
        
        for i, b in enumerate(self.decoder_blocks):
            print(b)
            print(x.shape)
            try: x = b(x)
            except: print(self.encoder_blocks[i+1])
            print(x.shape)
            print()
        return x
    
    def reparameterize(self, mu, var):
        pass
    
    def predict_from_image(self, image):
        pass
    
    def load_weights(self, path):
        path = path.replace('\\', '/')
        if path.endswith('/'):
            path = path[:-1]
        
        print(path)
        if path.endswith('.pth'):
            self.load_state_dict(torch.load(path))
            return int(path.split('/')[-1].split('_')[0]) + 1
        else:
            ckpts = glob.glob(path + "/*.pth")
            if len(ckpts) == 0: return 0
            ckpts = natsort.natsorted(ckpts)
            print(f"loading weights {ckpts[-1]}")
            
            self.load_state_dict(torch.load(ckpts[-1]))
            
            return int(os.path.split(ckpts[-1])[-1].split('_')[0]) + 1
    