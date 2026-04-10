import os
import torch
from torch.utils.data import Dataset
import h5py
import pywt
import numpy as np
import torchvision.transforms as transforms

class H5Dataset(Dataset):
    def __init__(self, h5_folder, dataset_name, mode='train'):
        self.mode = mode
        # Build filename based on mode and dataset name
        if mode == 'test':
            filename = f"{mode}_{dataset_name.lower()}_multiExm1.h5"
        else:
            filename = f"{mode}_{dataset_name.lower()}.h5"
        self.h5_path = os.path.join(h5_folder, filename)
        
        # Open file to check length (and keep handle if single thread, but better to open in getitem or use copy)
        # For simplicity and robustness with DataLoader num_workers, we often open in __getitem__
        # But we need the length now.
        with h5py.File(self.h5_path, 'r') as f:
            self.len = f['gt'].shape[0]
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            # Read data
            # Assuming data is stored as (N, C, H, W)
            lms = f['lms'][idx] # (8, 64, 64)
            pan = f['pan'][idx] # (1, 64, 64)
            gt = f['gt'][idx]   # (8, 64, 64)
            
        # Data is likely numpy array. Convert to tensor?
        # PanDataset uses transforms.ToTensor() which expects HWC [0, 255] or [0, 1]?
        # Usually H5 are float32 [0, 1] or uint8.
        # Let's assume float32 (C, H, W).
        # PanDataset doing `pywt.wavedec2(np.array(img_ms_up)...)` implies img_ms_up is PIL or HWC numpy.
        
        # If H5 is (C, H, W) numpy:
        # We need to adapt for pywt.wavedec2 which typically expects (H, W) or (H, W, C)?
        # pywt.wavedec2: "Input data... 2D ndarray"
        # It operates on 2D images.
        
        # PanDataset logic:
        # lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(np.array(img_ms_up), "db1", level=1, axes=[0,1])
        # output of pywt on (H, W, C) or (C, H, W)?
        # img_ms_up in PanDataset is PIL Image.resize -> np.array(PIL) is (H, W, C).
        # axes=[0, 1] means transform over H and W.
        
        # Our H5 data is likely (C, H, W).
        # We need to transpose to (H, W, C) for pywt if we use axes=[0, 1].
        
        lms_np = lms.transpose(1, 2, 0) # (H, W, C)
        pan_np = pan.transpose(1, 2, 0) # (H, W, 1)
        
        # Wavelet Transform
        # Note: PanDataset uses "db1", level=1, axes=[0,1]
        lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(lms_np, "db1", level=1, axes=[0,1])
        pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(pan_np, "db1", level=1, axes=[0,1])
        
        # Turn back to tensor
        # PanDataset uses self.to_tensor() which makes (H,W,C)->(C,H,W) and scales [0,255]->[0,1] if int.
        # IF H5 is already float [0, 1], we just need to verify range.
        # If H5 is already tensors we can just wrap.
        
        # Let's assume we return tensors float32.
        
        # Data is in range 0-2047 (WV3 11-bit), normalize to 0-1
        img_ms_up = torch.from_numpy(lms).float() / 2047.0
        img_pan = torch.from_numpy(pan).float() / 2047.0
        img_GT = torch.from_numpy(gt).float() / 2047.0
        
        # Wavelets results are numpy (H/2, W/2, C). Need to be (C, H/2, W/2).
        def to_tensor(x):
            # Normalize wavelets too?
            # Wavelets on normalized data is equiv to normalized wavelets? Yes (linearity).
            # But here we compute wavelets on un-normalized numpy then convert?
            # Or compute on normalized tensor?
            # Code uses pywt on numpy.
            # If we normalize lms_np/pan_np first:
            return torch.from_numpy(x.transpose(2, 0, 1)).float()

        # Normalize numpy before wavelets for consistency
        lms_np = lms_np / 2047.0
        pan_np = pan_np / 2047.0

        lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(lms_np, "db1", level=1, axes=[0,1])
        pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(pan_np, "db1", level=1, axes=[0,1])
        
        wavelets_dcp = torch.cat([
            to_tensor(lms_main), 
            to_tensor(pan_h), 
            to_tensor(pan_d), 
            to_tensor(pan_v)
        ], dim=0)
        
        return {
            'img_ms_up': img_ms_up, 
            'img_pan': img_pan, 
            'wavelets': wavelets_dcp,
            'GT': img_GT,
            'file_name': f"{idx}.png" # Dummy filename
        }
