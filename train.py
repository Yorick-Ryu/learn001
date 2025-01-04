import os
import torch
import numpy as np
# 添加内存优化配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.unet import UNet
from models.resnet import ResNetSegmentation
from datasets.loader import SegmentationDataset

def train(args):
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
    
    print(f"正在加载数据集，路径: {args.data_dir}")
    try:
        dataset = SegmentationDataset(args.dataset, args.data_dir)
        train_loader = dataset.get_loader(args.batch_size)
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        raise

    # 初始化tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # 选择并初始化模型
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=args.num_classes)
    else:
        # 为ResNet添加pretrained参数
        pretrained = not getattr(args, 'no_pretrained', False)
        model = ResNetSegmentation(n_classes=args.num_classes, pretrained=pretrained)
        
    # 加载预训练模型
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    # 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 验证标签分布
            unique_labels = torch.unique(target)
            valid_labels = unique_labels[unique_labels != 255]
            
            # 过滤掉全是忽略标签的样本
            if len(valid_labels) == 0:
                continue

            # 添加数据验证
            if torch.isnan(data).any() or torch.isnan(target).any():
                print(f"警告：数据包含NaN值，跳过此批次")
                continue
            
            # 添加标签值检查
            unique_labels = torch.unique(target)
            if args.dataset == 'camvid':
                valid_range = (unique_labels >= 0) & (unique_labels < 11) | (unique_labels == 255)
            else:
                valid_range = (unique_labels >= 0) & (unique_labels < 19) | (unique_labels == 255)
                
            if not valid_range.all():
                invalid_labels = unique_labels[~valid_range]
                print(f"警告：发现无效的标签值: {invalid_labels}")
                continue

            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 使用带ignore_index的损失函数
            criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
            loss = criterion(output, target)
            
            # 检查损失值
            if loss.item() == 0 or torch.isnan(loss):
                print(f"警告：损失值异常 - {loss.item()}")
                continue
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每10个批次打印一次进度
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}')
            
            # 每100个批次记录一张预测图
            if batch_idx % 100 == 0:
                # 获取预测结果
                with torch.no_grad():
                    pred = torch.argmax(output, dim=1)
                    
                    # 转换为RGB格式显示
                    input_img = data[0]  # CHW格式
                    
                    # 转换target为3通道图像
                    target_img = torch.zeros(3, target.size(1), target.size(2))
                    target_img[0] = target[0].float() / args.num_classes
                    target_img[1] = target[0].float() / args.num_classes
                    target_img[2] = target[0].float() / args.num_classes
                    
                    # 转换pred为3通道图像
                    pred_img = torch.zeros(3, pred.size(1), pred.size(2))
                    pred_img[0] = pred[0].float() / args.num_classes
                    pred_img[1] = pred[0].float() / args.num_classes
                    pred_img[2] = pred[0].float() / args.num_classes
                    
                    # 记录图像
                    writer.add_image('Image/input', input_img, global_step=epoch)
                    writer.add_image('Image/target', target_img, global_step=epoch)
                    writer.add_image('Image/pred', pred_img, global_step=epoch)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.6f}')
        
        # 记录到tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 记录学习率
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录模型参数的梯度直方图
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'grad/{name}', param.grad, epoch)
                writer.add_histogram(f'weight/{name}', param, epoch)
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 
                      f'{args.log_dir}/model_epoch_{epoch+1}.pth')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['unet', 'resnet'])
    parser.add_argument('--dataset', choices=['cityscapes', 'camvid'])
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=2)  # 修改默认批次大小
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--no-pretrained', action='store_true',
                      help='不使用预训练模型，使用随机初始化')
    
    args = parser.parse_args()
    train(args)
