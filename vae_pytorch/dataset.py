import glob, cv2, os

from torch.utils.data import Dataset

from utils.functions import preprocess_image, mat_to_tensor

class CustomDataset(Dataset):
    def __init__(self, root_dir, target_size, transform=None):
        
        self.root_dir = root_dir
        self.target_size = target_size
        
        self.img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir)
                          for f in filenames if os.path.splitext(f)[1] in ['.jpg', '.png', 'jpeg']]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = preprocess_image(img, self.target_size)
        img = mat_to_tensor(img)
        
        return img
    