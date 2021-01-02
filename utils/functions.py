import torch
import cv2
import numpy as np

def preprocess_image(img, target_size):
    
    if type(target_size) == int:
        target_size = (target_size, target_size)
    elif type(target_size) == list:
        target_size = tuple(target_size)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=target_size)
    img = img / 255.
    
    return img

def postprocess_image(img):
    
    img = img * 255
    
    return img.astype(np.uint8)

def mat_to_tensor(mat):
    
    mat = mat.astype(np.float64)
    mat = mat.transpose((2, 0, 1))
    mat = torch.Tensor(mat)
    
    return mat

def tensor_to_mat(tensor):
    
    mat = tensor.detach().numpy()
    mat = mat.transpose((0, 2, 3, 1))
    
    return mat
