

import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
class Classifier(nn.Module):
    def __init__(self, nums_frames):
        super(Classifier, self).__init__();
        self.nums_frames = nums_frames
        # 假设使用卷积层来处理空间维度
        self.conv = nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # 全连接层
        self.fc1 = nn.Linear(512 * (nums_frames // 4) * 3 * 3, 256)
        self.fc2 = nn.Linear(256, nums_frames)

    def forward(self, x):
        # 卷积层
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        # 展平张量
        x = x.view(-1, 512 * (self.nums_frames // 4) * 3 * 3)
        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# class Classifier(nn.Module):
#     def __init__(self, nums_frames):
#         super(Classifier, self).__init__()
#         self.nums_frames = nums_frames
#         self.conv = nn.Conv3d(1024, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
#         self.fc1 = nn.Linear(512 * (nums_frames // 4) * 3 * 3, 256)
#         self.fc2 = nn.Linear(256, nums_frames)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(-1, 512 * (self.nums_frames // 4) * 3 * 3)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)  # 添加Sigmoid激活
#         return x
    
# 模型的定义
class SwinTransformer3DWithHead(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, patch_size, window_size, drop_path_rate, patch_norm,num_frames):
        super(SwinTransformer3DWithHead, self).__init__()
        self.backbone = SwinTransformer3D(embed_dim=embed_dim, 
                                          depths=depths, 
                                          num_heads=num_heads, 
                                          patch_size=patch_size, 
                                          window_size=window_size, 
                                          drop_path_rate=drop_path_rate, 
                                          patch_norm=patch_norm)
        self.classifier = Classifier(num_frames)

    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits