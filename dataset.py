import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from natsort import natsorted
from torchvision import transforms
from torchvision.io import read_image
import random
class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train_txt_path,test_txt_path, flag="eval"):
        """
        :param img_dir: 图像路径
        :param mask_dir: 存放真实标签的 .npy 文件路径
        :param transform: 图像变换（如有）
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        
        # 读取所有 npy 文件，存入字典
        self.labels = {}
        for npy_file in os.listdir(mask_dir):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(mask_dir, npy_file)
                self.labels[npy_file] = np.load(file_path)
        ## 弱监督数据集文件读取
        train_path_list = []
        test_path_list = []
        with open(train_txt_path, 'r') as f:
            for path in f.readlines():
                train_path_list.append(path.replace("\n",""))
        with open(test_txt_path, 'r') as f:
            for path in f.readlines():
                test_path_list.append(path.replace("\n",""))
        root_dir1 = os.path.join(img_dir, 'train')
        root_dir2 = os.path.join(img_dir, 'test')
        # 获取所有图片路径和 labels 
        if flag == "train":

            self.frames_path_group,self.labels_group = self.get_frames_images_labels(train_path_list,root_dir1,root_dir2,mask_dir)
        elif flag == "eval":
            ## 采样 1/10 验证
            self.frames_path_group,self.labels_group = self.get_frames_images_labels(test_path_list,root_dir1,root_dir2,mask_dir)
            self.frames_path_group,self.labels_group = self.sample_subset()
        else :
            self.frames_path_group,self.labels_group = self.get_frames_images_labels(test_path_list,root_dir1,root_dir2,mask_dir)

        
        print(len(self.frames_path_group),len(self.labels_group))
        ### transformer
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到模型输入大小
        transforms.ToTensor(),  # 转换为张量 (C, H, W)，值范围 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                            std=[0.229, 0.224, 0.225])   # ImageNet 标准差
        ])


    def sample_subset(self):
        random.seed(42)
        num_samples = len(self.frames_path_group)
        sample_indices = random.sample(range(num_samples), num_samples // 10)
        frames_subset = [self.frames_path_group[i] for i in sample_indices]
        labels_subset = [self.labels_group[i] for i in sample_indices]
        return frames_subset, labels_subset
    def get_frames_images_labels(self,path_list,root_dir1,root_dir2,mask_dir,length=8):
        frames_group = []
        frames_labels = []

        for video_id in path_list:
            if os.path.exists(os.path.join(root_dir1, video_id)):
                video_path = os.path.join(root_dir1, video_id)
                npy_file = np.zeros(len(os.listdir(video_path)),dtype=np.float32)  ### 确定是正常的是 0 吗 ？ 
            else :
                video_path = os.path.join(root_dir2, video_id)
                npy_path = os.path.join(mask_dir, video_id+".npy")
                npy_file = np.load(npy_path).astype(np.float32)
            
            # 获取所有图片并按名字排序
            images = natsorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            
            # 遍历两次分别获取偶数和奇数索引图片
            even_images = images[::2]  # 偶数索引图片
            odd_images = images[1::2]  # 奇数索引图片
            even_labels = npy_file[::2]
            odd_labels = npy_file[1::2]
            assert len(even_images) == len(even_labels)
            assert len(odd_images) == len(odd_labels)  ## z这两句话基本可以断定数据集与labels是加载对的，是对应的
            for i in range(0, len(even_images), length): ## 确保最后不足length的维度 + length 后不会超过长度
                group = even_images[i:i+length]
                if len(group) < length:
                    continue  ## 不足 length 的不要训练
                frames_group.append([os.path.join(video_path, img) for img in group])
                frames_labels.append(even_labels[i:i+length])
            
            for i in range(0, len(odd_images), length):
                group = odd_images[i:i+length]
                if len(group) < length:
                    continue  ## 不足 length 的不要训练
                frames_group.append([os.path.join(video_path, img) for img in group])
                frames_labels.append(odd_labels[i:i+length])
        return frames_group,frames_labels
    # print(result)
    # break
    def __len__(self):
        return len(self.frames_path_group)

    def __getitem__(self, idx):
        a_frames_group , labels  = self.frames_path_group[idx],self.labels_group[idx]
 
        # 读取图片
        a_frames_group = [self.transform(Image.open(frame_path))for frame_path in a_frames_group]
        # 堆叠为 (frames, C, H, W)
        a_frames_group = torch.stack(a_frames_group)  # (T, C, H, W)
        # 调整维度为 (C, T, H, W)
        a_frames_group = a_frames_group.permute(1, 0, 2, 3)  # (C, T, H, W)
        return a_frames_group, labels


# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import os
# from PIL import Image
# from natsort import natsorted
# from torchvision import transforms
# from torchvision.io import read_image
# import random
# class ShanghaiTechDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, train_txt_path,test_txt_path, is_train=True,is_val=True):
#         """
#         :param img_dir: 图像路径
#         :param mask_dir: 存放真实标签的 .npy 文件路径
#         :param transform: 图像变换（如有）
#         """
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir

        
#         # 读取所有 npy 文件，存入字典
#         self.labels = {}
#         for npy_file in os.listdir(mask_dir):
#             if npy_file.endswith('.npy'):
#                 file_path = os.path.join(mask_dir, npy_file)
#                 self.labels[npy_file] = np.load(file_path)
#         ## 弱监督数据集文件读取
#         train_path_list = []
#         test_path_list = []
#         with open(train_txt_path, 'r') as f:
#             for path in f.readlines():
#                 train_path_list.append(path.replace("\n",""))
#         with open(test_txt_path, 'r') as f:
#             for path in f.readlines():
#                 test_path_list.append(path.replace("\n",""))
#         root_dir1 = os.path.join(img_dir, 'train')
#         root_dir2 = os.path.join(img_dir, 'test')
#         # 获取所有图片路径和 labels 
#         if is_train == True :
#             self.frames_path_group,self.labels_group = self.get_frames_images_labels(train_path_list,root_dir1,root_dir2,mask_dir)
#         elif is_val  == True : ## 采样 1/10 验证
#             self.frames_path_group,self.labels_group = self.get_frames_images_labels(test_path_list,root_dir1,root_dir2,mask_dir)
#             self.frames_path_group,self.labels_group = self.sample_subset()
#             print("采样成功")
#         else:
#             self.frames_path_group,self.labels_group = self.get_frames_images_labels(test_path_list,root_dir1,root_dir2,mask_dir)
#         print("is_val",is_val)
        
#         print(len(self.frames_path_group),len(self.labels_group))
#         ### transformer
#         self.transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # 调整到模型输入大小
#         transforms.ToTensor(),  # 转换为张量 (C, H, W)，值范围 [0,1]
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
#                             std=[0.229, 0.224, 0.225])   # ImageNet 标准差
#         ])


#     def sample_subset(self):
#         random.seed(42)
#         num_samples = len(self.frames_path_group)
#         sample_indices = random.sample(range(num_samples), num_samples // 10)
#         frames_subset = [self.frames_path_group[i] for i in sample_indices]
#         labels_subset = [self.labels_group[i] for i in sample_indices]
#         return frames_subset, labels_subset
#     def get_frames_images_labels(self,path_list,root_dir1,root_dir2,mask_dir,length=8):
#         frames_group = []
#         frames_labels = []

#         for video_id in path_list:
#             if os.path.exists(os.path.join(root_dir1, video_id)):
#                 video_path = os.path.join(root_dir1, video_id)
#                 npy_file = np.zeros(len(os.listdir(video_path)),dtype=np.float32)  ### 确定是正常的是 0 吗 ？ 
#             else :
#                 video_path = os.path.join(root_dir2, video_id)
#                 npy_path = os.path.join(mask_dir, video_id+".npy")
#                 npy_file = np.load(npy_path).astype(np.float32)
            
#             # 获取所有图片并按名字排序
#             images = natsorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
            
#             # 遍历两次分别获取偶数和奇数索引图片
#             even_images = images[::2]  # 偶数索引图片
#             odd_images = images[1::2]  # 奇数索引图片
#             even_labels = npy_file[::2]
#             odd_labels = npy_file[1::2]
#             assert len(even_images) == len(even_labels)
#             assert len(odd_images) == len(odd_labels)  ## z这两句话基本可以断定数据集与labels是加载对的，是对应的
#             for i in range(0, len(even_images), length): ## 确保最后不足length的维度 + length 后不会超过长度
#                 group = even_images[i:i+length]
#                 if len(group) < length:
#                     continue  ## 不足 length 的不要训练
#                 frames_group.append([os.path.join(video_path, img) for img in group])
#                 frames_labels.append(even_labels[i:i+length])
            
#             for i in range(0, len(odd_images), length):
#                 group = odd_images[i:i+length]
#                 if len(group) < length:
#                     continue  ## 不足 length 的不要训练
#                 frames_group.append([os.path.join(video_path, img) for img in group])
#                 frames_labels.append(odd_labels[i:i+length])
#         return frames_group,frames_labels
#     # print(result)
#     # break
#     def __len__(self):
#         return len(self.frames_path_group)

#     def __getitem__(self, idx):
#         a_frames_group , labels  = self.frames_path_group[idx],self.labels_group[idx]
 
#         # 读取图片
#         a_frames_group = [self.transform(Image.open(frame_path))for frame_path in a_frames_group]
#         # 堆叠为 (frames, C, H, W)
#         a_frames_group = torch.stack(a_frames_group)  # (T, C, H, W)
#         # 调整维度为 (C, T, H, W)
#         a_frames_group = a_frames_group.permute(1, 0, 2, 3)  # (C, T, H, W)
#         return a_frames_group, labels
