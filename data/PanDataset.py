import os

import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.multiprocessing
import numpy as np
import pywt
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

torch.multiprocessing.set_sharing_strategy('file_system')

# 获取路径
py_path = os.path.abspath(__file__) 
file_dir = os.path.dirname(py_path) 

class PanDataset(Dataset):
    def __init__(self, img_size, ms_folder, pan_folder, ms_is_GT=True, mode='train', start_iter=0):
        self.images = [f for f in os.listdir(ms_folder) if f.endswith('.tif')]
        self.ms_folder = ms_folder
        self.pan_folder = pan_folder
        self.mode = mode
        self.img_size = img_size
        self.ms_is_GT = ms_is_GT    
        self.iter = start_iter
        
        # if mode == 'train':
        #     self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        #     self.random_vertical_flip = transforms.RandomVerticalFlip(p=0.5)
        #     self.random_crop = transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC)
        #     self.random_rot = transforms.RandomChoice([
        #         transforms.RandomRotation((0, 0)),
        #         transforms.RandomRotation((90, 90)),
        #         transforms.RandomRotation((180, 180)),
        #         transforms.RandomRotation((270, 270)),
        #     ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def seed_everything(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def random_transform(self, img, seed):
        self.seed_everything(seed)
        img = self.random_horizontal_flip(img)
        img = self.random_vertical_flip(img)
        img = self.random_rot(img)
        if self.iter < 100_000:
            img = self.random_crop(img)
        
        return img
    
    def __getitem__(self, idx):
        self.iter += 1
        # 加载图像
        ms_image_path = os.path.join(self.ms_folder, self.images[idx])
        pan_image_path = os.path.join(self.pan_folder, self.images[idx]) 

        img_ms = Image.open(ms_image_path)
        img_pan = Image.open(pan_image_path)
        
        # 训练模式下进行一致的数据增强
        # if self.mode == 'train':
        #     seed = np.random.randint(2147483647) 

        #     img_ms = self.random_transform(img_ms, seed)
        #     img_pan = self.random_transform(img_pan, seed)
            
        if self.ms_is_GT:
            img_GT = img_ms
            img_ms_up = img_ms.resize((self.img_size // 4, self.img_size // 4), Image.BICUBIC)
            img_ms_up = img_ms_up.resize((self.img_size, self.img_size), Image.BICUBIC)
        else:
            img_ms_up = img_ms.resize((self.img_size, self.img_size), Image.BICUBIC)
            img_GT = img_ms_up
            
        # Normalize to [0, 1] (Assuming WV3 11-bit data)
        # Convert PIL to numpy float32 and normalize
        img_ms_up_np = np.array(img_ms_up).astype(np.float32) / 2047.0
        img_pan_np = np.array(img_pan).astype(np.float32) / 2047.0
        img_GT_np = np.array(img_GT).astype(np.float32) / 2047.0
            
        lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(img_ms_up_np, "db1", level=1, axes=[0,1])
        pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(img_pan_np, "db1", level=1, axes=[0,1])
        
        img_ms_up = torch.from_numpy(img_ms_up_np).permute(2, 0, 1) # (H,W,C) -> (C,H,W)
        img_pan = torch.from_numpy(img_pan_np).unsqueeze(0) # (H,W) -> (1,H,W) or (H,W,1)->(1,H,W)?
        # Check PIL mode. If 'I', numpy is (H,W). If RGB, (H,W,3).
        # Assuming Pan is single channel.
        if len(img_pan_np.shape) == 2:
             img_pan = torch.from_numpy(img_pan_np).unsqueeze(0)
        else:
             img_pan = torch.from_numpy(img_pan_np).permute(2, 0, 1)
             
        img_GT = torch.from_numpy(img_GT_np).permute(2, 0, 1)

        wavelets_dcp = torch.cat([self.to_tensor(lms_main).float(), self.to_tensor(pan_h).float(), 
                            self.to_tensor(pan_d).float(), self.to_tensor(pan_v).float()], dim=0)
        return {
            'img_ms_up': img_ms_up, 'img_pan': img_pan, 'wavelets': wavelets_dcp,
            'GT': img_GT,
            'file_name': self.images[idx]
        }

if __name__ == '__main__':
    ms_folder = r"E:\code\PanSharpening\PanDataset\WV3_data\train128\ms"
    pan_folder = r"E:\code\PanSharpening\PanDataset\WV3_data\train128\pan"
    img_size = 128
    dataset = PanDataset(img_size, ms_folder, pan_folder, ms_is_GT=True, mode='train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    for data in dataloader:
        print(data['img_ms_up'].shape)
        print(data['img_pan'].shape)
        print(data['wavelets'].shape)
        print(data['GT'].shape)

