import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from dataset import ShanghaiTechDataset
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
# 设备选择，使用所有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from model import SwinTransformer3DWithHead
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using multiple GPUs")

    # 数据相关
    parser.add_argument('--img_dir', type=str, default=r"../vad_datasets/frames", help="Directory of the image data")
    parser.add_argument('--mask_dir', type=str, default=r"../vad_datasets/test_frame_mask",  help="Directory of the mask data")
    parser.add_argument('--train_txt_path', type=str,default=r"../vad_datasets/SH_Train.txt",help="Path to the train txt file")
    parser.add_argument('--test_txt_path', type=str, default=r"../vad_datasets/SH_Test.txt", help="Path to the test txt file")

    # 模型相关
    # parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    # parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 18, 2], help="Depths for each stage of the model")
    # parser.add_argument('--num_heads', type=int, nargs='+', default=[4, 8, 16, 32], help="Number of attention heads for each stage")
    # parser.add_argument('--patch_size', type=tuple, default=(2, 4, 4), help="Patch size")
    # parser.add_argument('--window_size', type=tuple, default=(16, 7, 7), help="Window size")
    # parser.add_argument('--drop_path_rate', type=float, default=0.4, help="Drop path rate")
    # parser.add_argument('--num_frames', type=int, default=8, help="Number of frames")
    
    # 训练相关
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per GPU")
    parser.add_argument('--is_train', action='store_true', help="true is train or  default test")
    parser.add_argument('--is_eval', action='store_true', help="save and eval")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save_interval', type=int, default=2, help="Interval (in epochs) to save the model")
    parser.add_argument('--save_path', type=str, default='./checkpoints', help="Path to save the model checkpoints")

    # 其他
    parser.add_argument('--pretrained_weights', type=str, default='checkpoints/swin_epoch_30.pth', help="Path to the pretrained weights")
    parser.add_argument('--gpu', type=str, default="0", help="GPU device number(s) to use (default: '0', can specify multiple GPUs, e.g. '0,1', or '-1' for CPU)")
    return parser.parse_args()

# 设备选择，支持 CPU 或多个 GPU
def setup_device(args):
    if args.gpu == "-1":  # 使用 CPU
        device = torch.device("cpu")
        print("Using CPU")
    else:  # 使用 GPU，可能是单个或多个
        gpus = args.gpu.split(",")  # 解析多个 GPU 号
        device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")  # 默认选择第一个 GPU

        if len(gpus) > 1:  # 如果指定了多个 GPU
            print(f"Using multiple GPUs: {gpus}")
            # 设置 `DataParallel`，并在多个 GPU 上运行
            device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        else:
            print(f"Using GPU: {gpus[0]}")
        
    return device
# 加载模型和预训练权重

# 加载预训练模型
def load_model(args):
    model = SwinTransformer3DWithHead(embed_dim=128, 
                                    depths=[2, 2, 18, 2], 
                                    num_heads=[4, 8, 16, 32], 
                                    patch_size=(2, 4, 4), 
                                    window_size=(16, 7, 7), 
                                    drop_path_rate=0.4, 
                                    patch_norm=True,
                                    num_frames=8)
    
    checkpoint = torch.load(args.pretrained_weights)
    # print(checkpoint.keys())
    # 去掉 module. 前缀
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace('module.', '')  # 去掉 module. 前缀
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    print("加载模型")
    
    # 如果使用多个 GPU，启用 DataParallel
    if len(args.gpu.split(",")) > 1 and torch.cuda.is_available():
        print(f"Using {len(args.gpu.split(','))} GPUs for DataParallel")
        model = nn.DataParallel(model)

    return model

# 创建数据集
def create_dataloader(args):
    train_dataset = ShanghaiTechDataset(img_dir=args.img_dir, 
                                        mask_dir=args.mask_dir, 
                                        train_txt_path=args.train_txt_path,
                                        test_txt_path=args.test_txt_path, 
                                        is_train=args.is_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.is_train)
    return train_loader

# 训练模型
def test(args, model, test_loader, device):
    # 使用 tqdm 包装 train_loader
    progress_bar = tqdm(test_loader)
    targets = []
    preds = []
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        

        outputs = model(inputs).view(-1).cpu().detach().numpy()
        labels_numpy = labels.view(-1).cpu().detach().numpy()
        targets = targets + list(labels_numpy)
        preds = preds + list(outputs)

        # loss = criterion(outputs, labels)
        # running_loss += loss.item()
        # progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    # 示例数据
    print(targets)
    print(preds)
    auc_result = calculate_auc(targets, preds)
    print(f"AUC: {auc_result}")

def calculate_auc(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    auc = roc_auc_score(target, pred)
    return auc

def main():
    # 解析命令行参数
    print()
    args = parse_args()

    # 设备选择
    device = setup_device(args)

    # 加载模型
    model = load_model(args)
    model.to(device)

    # 创建数据加载器
    train_loader = create_dataloader(args)

    # 训练模型
    test(args, model, train_loader, device)

if __name__ == "__main__":
    main()
