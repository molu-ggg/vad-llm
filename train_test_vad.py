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
# 设备选择，使用所有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

from model import SwinTransformer3DWithHead
import numpy as np 
def calculate_auc(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    auc = roc_auc_score(target, pred)
    return auc
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using multiple GPUs")

    # 数据相关
    # /mydata/AST/STG-NF-main/data/ShanghaiTech/gt/test_frame_mask
    #/mydata/AST/STG-NF-main/data/ShanghaiTech/images/test
    parser.add_argument('--is_train', action='store_true', help="true is train or  default test")
    parser.add_argument('--is_eval', action='store_true', help="save and eval")
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
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size per GPU")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--save_interval', type=int, default=2, help="Interval (in epochs) to save the model")
    parser.add_argument('--save_path', type=str, default='./checkpoints', help="Path to save the model checkpoints")

    # 其他
    parser.add_argument('--checkpoint', type=str, default='checkpoints/swin_base.pth', help="Path to the pretrained weights")
    parser.add_argument('--pretrained_weights', type=str, default='checkpoints/swin_base_patch244_window1677_sthv2.pth', help="Path to the pretrained weights")
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
    if args.is_train:
        print("加载预训练模型")
        checkpoint = torch.load(args.pretrained_weights)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]  # 去除 'backbone.' 前缀
                new_state_dict[name] = v
        model.backbone.load_state_dict(new_state_dict)
    else:
        print("加载模型待测试")
        checkpoint = torch.load(args.checkpoint)       
        model.load_state_dict(new_state_dict)
    
    
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
                                        flag="train")
    eval_dataset = ShanghaiTechDataset(img_dir=args.img_dir, 
                                    mask_dir=args.mask_dir, 
                                    train_txt_path=args.train_txt_path,
                                    test_txt_path=args.test_txt_path, 
                                    flag="eval")
    test_dataset= ShanghaiTechDataset(img_dir=args.img_dir, 
                                    mask_dir=args.mask_dir, 
                                    train_txt_path=args.train_txt_path,
                                    test_txt_path=args.test_txt_path, 
                                    flag="test")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,eval_loader,test_loader
# 训练模型
def test(args, model, test_loader, device):
    # 使用 tqdm 包装 train_loader
    progress_bar = tqdm(test_loader)
    targets = []
    preds = []
    model.eval()
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
    return auc_result
# 训练模型
def train(args, model, train_loader, eval_loader,test_loader,device):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([154174 / 9364], dtype=torch.float32).to(device))  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)

    print("开始训练")
    auc_result = []
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        # 使用 tqdm 包装 train_loader
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs,labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 在进度条中显示当前损失
            progress_bar.set_postfix(loss=loss.item())

        # 平均损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")

        eval_auc = test(args, model, eval_loader, device)
        auc_result.append([epoch+1,eval_auc])
        print(epoch+1, "--AUC:", eval_auc)

        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_file = os.path.join(args.save_path, f"swin_epoch_{epoch+1}.pth")
            # torch.save(model.state_dict(), save_file)
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }

            torch.save(checkpoint, save_file)
            print(f"模型已保存到: {save_file}")


    test_auc = test(args, model, eval_loader, device)
    print(auc_result)
    print("final AUC:",test_auc)
    

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
    train_loader,eval_loader,test_loader = create_dataloader(args)

    # 训练模型
    train(args, model, train_loader,eval_loader,test_loader, device)
    # test(args, model, train_loader, device)


if __name__ == "__main__":
    main()
# python train_vad.py --img_dir /mydata/AST/STG-NF-main/data/ShanghaiTech/images/ --mask_dir /mydata/AST/STG-NF-main/data/ShanghaiTech/gt/test_frame_mask --train_txt_path /mydata/ygq/SH_Train.txt --test_txt_path /mydata/ygq/SH_Test.txt
# scp -r -P 3333 /home/data/agqing/vad_datasets/frames/test/01_0026 lancer@10.26.61.39:/mydata/AST/STG-NF-main/data/ShanghaiTech/images/test/01_0026