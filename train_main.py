#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import warnings
import re

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description='双模态三分类训练脚本（改进损失函数版）')
    parser.add_argument('--bases_root', required=True,
                      help='bases图像根目录，其下是三个类别文件夹')
    parser.add_argument('--cigar_root', required=True,
                      help='cigar图像根目录，其下是三个类别文件夹')
    parser.add_argument('--class_dirs', nargs=3, required=True,
                      metavar=('CLS0', 'CLS1', 'CLS2'),
                      help='三个类别文件夹名，顺序=label 0/1/2')
    parser.add_argument('--train_chrs', nargs='+', default=[f'chr_{i}' for i in range(1, 13)],
                      help='用于训练集的染色体名称，如chr_1 chr_2 ... chr_12')
    parser.add_argument('--test_chrs', nargs='+', default=['chr_13', 'chr_14', 'chr_15', 'chr_21', 'chr_22', 'chr_X', 'chr_Y'],
                      help='用于测试集的染色体名称，如chr_13 chr_14 ... chr_Y')

    parser.add_argument('--model', default='resnet50',
                      choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='交叉熵损失权重(0-1)，对比损失权重为1-alpha')
    parser.add_argument('--temp', type=float, default=0.1,
                      help='对比损失的温度系数')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', default='best_model.pth')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                      help='GPU card numbers to use, e.g., --gpu 0 1 (default: [0])')

    return parser.parse_args()

# ---------- 工具 ----------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, val_tf

def extract_chr_from_filename(filename):
    """从文件名中提取染色体信息"""
    match = re.search(r'(chr_[0-9XY]+)', filename)
    return match.group(1) if match else None

def extract_id_from_filename(filename):
    """从文件名中提取唯一标识符（用于对齐bases和cigar图像）"""
    match = re.search(r'(chr_[0-9XY]+_\d+_[A-Z]+_\d+)', filename)
    return match.group(1) if match else None

class DualModalDataset(Dataset):
    def __init__(self, bases_root, cigar_root, class_dirs, chr_list, transform=None):
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        self.samples = []
        
        # 收集bases图像路径
        bases_samples = []
        for cls_name in class_dirs:
            cls_dir = os.path.join(bases_root, cls_name)
            for file in os.listdir(cls_dir):
                if file.endswith('.png'):
                    chr_name = extract_chr_from_filename(file)
                    if chr_name in chr_list:
                        file_id = extract_id_from_filename(file)
                        if file_id:
                            bases_samples.append((
                                os.path.join(cls_dir, file),
                                self.class_to_idx[cls_name],
                                file_id
                            ))
        
        # 收集cigar图像路径并匹配
        cigar_samples = {}
        for cls_name in class_dirs:
            cls_dir = os.path.join(cigar_root, cls_name)
            for file in os.listdir(cls_dir):
                if file.endswith('.png'):
                    file_id = extract_id_from_filename(file)
                    if file_id:
                        cigar_samples[file_id] = os.path.join(cls_dir, file)
        
        # 对齐样本
        for bases_path, label, file_id in bases_samples:
            if file_id in cigar_samples:
                self.samples.append((
                    bases_path,
                    cigar_samples[file_id],
                    label
                ))
        
        print(f"Found {len(self.samples)} aligned samples for {'train' if 'chr_1' in chr_list else 'test'} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bases_path, cigar_path, label = self.samples[idx]
        
        bases_img = datasets.folder.default_loader(bases_path)
        cigar_img = datasets.folder.default_loader(cigar_path)
        
        if self.transform:
            bases_img = self.transform(bases_img)
            cigar_img = self.transform(cigar_img)
        
        return (bases_img, cigar_img), label

# 双交叉注意力模块
class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query1 = nn.Linear(feature_dim, feature_dim)
        self.key1 = nn.Linear(feature_dim, feature_dim)
        self.value1 = nn.Linear(feature_dim, feature_dim)
        
        self.query2 = nn.Linear(feature_dim, feature_dim)
        self.key2 = nn.Linear(feature_dim, feature_dim)
        self.value2 = nn.Linear(feature_dim, feature_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, feat1, feat2):
        # 第一个模态的注意力
        q1 = self.query1(feat1)
        k1 = self.key1(feat1)
        v1 = self.value1(feat1)
        
        # 第二个模态的注意力
        q2 = self.query2(feat2)
        k2 = self.key2(feat2)
        v2 = self.value2(feat2)
        
        # 交叉注意力计算
        attn1 = self.softmax(torch.bmm(q1, k2.transpose(1, 2)))
        out1 = torch.bmm(attn1, v2)
        
        attn2 = self.softmax(torch.bmm(q2, k1.transpose(1, 2)))
        out2 = torch.bmm(attn2, v1)
        
        # 残差连接
        out1 = self.gamma1 * out1 + feat1
        out2 = self.gamma2 * out2 + feat2
        
        # 特征融合
        fused_feat = torch.cat([out1, out2], dim=-1)
        return fused_feat

class DualModalLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=0.1):
        super().__init__()
        self.alpha = alpha
        self.temp = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, model_output, targets, features1, features2):
        # 1. 标准交叉熵损失
        ce_loss = self.ce_loss(model_output, targets)
        
        # 2. 模态对比损失
        batch_size = features1.size(0)
        
        # 归一化特征向量
        features1 = nn.functional.normalize(features1, dim=1)
        features2 = nn.functional.normalize(features2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features1, features2.T) / self.temp
        
        # 创建对比学习目标
        labels = torch.arange(batch_size).to(features1.device)
        
        # 对称对比损失
        logits_12 = sim_matrix
        logits_21 = sim_matrix.T
        
        contrast_loss = (
            nn.functional.cross_entropy(logits_12, labels) + 
            nn.functional.cross_entropy(logits_21, labels)
        ) / 2
        
        # 3. 总损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * contrast_loss
        
        return total_loss, ce_loss, contrast_loss

class DualModalModel(nn.Module):
    def __init__(self, base_model, num_classes, device_ids=None):
        super().__init__()
        # 两个独立的ResNet特征提取器
        self.bases_backbone = build_model(base_model, num_classes=0, device_ids=device_ids)
        self.cigar_backbone = build_model(base_model, num_classes=0, device_ids=device_ids)
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.rand(1, 3, 224, 224)
            if base_model == 'resnet50':
                feature_dim = 2048
            else:
                feature_dim = 512
        
        # 交叉注意力融合模块
        self.cross_attn = CrossAttentionFusion(feature_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x1, x2):
        # 提取两个模态的特征
        feat1 = self.bases_backbone(x1)  # [bs, feature_dim]
        feat2 = self.cigar_backbone(x2)   # [bs, feature_dim]
        
        # 保存原始特征用于对比损失
        raw_feat1, raw_feat2 = feat1, feat2
        
        # 添加序列维度用于注意力计算
        feat1 = feat1.unsqueeze(1)  # [bs, 1, feature_dim]
        feat2 = feat2.unsqueeze(1)
        
        # 交叉注意力融合
        fused_feat = self.cross_attn(feat1, feat2)
        fused_feat = fused_feat.squeeze(1)
        
        # 分类
        out = self.classifier(fused_feat)
        
        return out, raw_feat1, raw_feat2

def build_model(name, num_classes=0, device_ids=None):
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(name)
    
    # 移除最后的全连接层
    if num_classes > 0:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # 仅作为特征提取器
        model.fc = nn.Identity()
    
    # 多GPU处理
    if device_ids and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    return model

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_ce, running_contrast = 0.0, 0.0, 0.0
    all_y, all_pred = [], []
    
    for (x1, x2), y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        out, feat1, feat2 = model(x1, x2)
        
        # 计算损失
        total_loss, ce_loss, contrast_loss = criterion(out, y, feat1, feat2)
        
        running_loss += total_loss.item() * x1.size(0)
        running_ce += ce_loss.item() * x1.size(0)
        running_contrast += contrast_loss.item() * x1.size(0)
        
        pred = out.argmax(1)
        all_y.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())

    # 计算指标
    cm = confusion_matrix(all_y, all_pred, labels=[0, 1, 2])
    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    eps = 1e-7
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * TP / (2 * TP + FP + FN + eps)

    metrics = {
        'loss': running_loss / len(loader.dataset),
        'ce_loss': running_ce / len(loader.dataset),
        'contrast_loss': running_contrast / len(loader.dataset),
        'accuracy': accuracy.mean(),
        'recall': recall.mean(),
        'f1': f1.mean(),
        'cm': cm,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_ce, running_contrast = 0.0, 0.0, 0.0
    n = 0
    all_y, all_pred = [], []
    
    for (x1, x2), y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        
        out, feat1, feat2 = model(x1, x2)
        
        total_loss, ce_loss, contrast_loss = criterion(out, y, feat1, feat2)
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * x1.size(0)
        running_ce += ce_loss.item() * x1.size(0)
        running_contrast += contrast_loss.item() * x1.size(0)
        n += x1.size(0)
        
        pred = out.argmax(1)
        all_y.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
    
    metrics = {
        'loss': running_loss / n,
        'ce_loss': running_ce / n,
        'contrast_loss': running_contrast / n,
        'acc': accuracy_score(all_y, all_pred),
        'f1': f1_score(all_y, all_pred, average='macro'),
        'pre': precision_score(all_y, all_pred, average='macro')
    }
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备设置
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu[0]}')
        print(f'Using GPUs: {args.gpu}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_tf, val_tf = build_transforms()
    
    # 创建数据集
    train_ds = DualModalDataset(
        args.bases_root, args.cigar_root, args.class_dirs, args.train_chrs, train_tf)
    val_ds = DualModalDataset(
        args.bases_root, args.cigar_root, args.class_dirs, args.test_chrs, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)

    # 构建双模态模型
    model = DualModalModel(args.model, num_classes=3, device_ids=args.gpu if len(args.gpu) > 1 else None)
    model = model.to(device)

    # 使用改进的损失函数
    criterion = DualModalLoss(alpha=args.alpha, temperature=args.temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_acc = 0.0
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练阶段
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 学习率调整
        scheduler.step(val_metrics['accuracy'])
        
        epoch_time = time.time() - epoch_start

        # 打印训练信息
        print(f'\nEpoch {epoch:02d}/{args.epochs} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_metrics["loss"]:.4f} (CE: {train_metrics["ce_loss"]:.4f} Contrast: {train_metrics["contrast_loss"]:.4f})')
        print(f'Train ACC: {train_metrics["acc"]:.4f} | PRE: {train_metrics["pre"]:.4f} | F1: {train_metrics["f1"]:.4f}')
        print(f'Val   Loss: {val_metrics["loss"]:.4f} (CE: {val_metrics["ce_loss"]:.4f} Contrast: {val_metrics["contrast_loss"]:.4f})')
        print(f'ConfusionMatrix:\n{val_metrics["cm"]}')
        for i in range(3):
            print(f'Class {i}: TP={val_metrics["TP"][i]} FP={val_metrics["FP"][i]} TN={val_metrics["TN"][i]} FN={val_metrics["FN"][i]}')
        print(f'Macro ACC: {val_metrics["accuracy"]:.4f}')
        print(f'Macro REC: {val_metrics["recall"]:.4f}')
        print(f'Macro F1 : {val_metrics["f1"]:.4f}')

        # 保存最佳模型
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, args.save_path)
            print(f'Model saved with best val acc: {best_acc:.4f}')

    total_time = time.time() - total_start
    print(f'\nTraining completed in {total_time:.2f} seconds')
    print(f'Best validation accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()