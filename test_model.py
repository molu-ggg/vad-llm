import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D

# import torch
import torch.nn as nn
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3D

model = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)

# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py
checkpoint = torch.load('/home/data/agqing/video-swin-transformer-pytorch-main/checkpoints/swin_base_patch244_window1677_sthv2.pth')

new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    if 'backbone' in k:
        name = k[9:]
        new_state_dict[name] = v 

model.load_state_dict(new_state_dict) 

dummy_x = torch.rand(1, 3, 32, 224, 224)
logits = model(dummy_x)
print(logits.shape)