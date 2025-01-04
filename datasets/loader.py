import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class SegmentationDataset(Dataset):
    def __init__(self, dataset_name, data_dir, split='train'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        
        # 设置图像转换，减小尺寸到256x256
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 降低分辨率
            transforms.ToTensor(),
        ])
        
        # 修改标签转换，移除ToTensor
        self.target_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
        # Cityscapes标签ID映射
        self.id_to_trainid = {
            0: 255,  # unlabeled
            1: 255,  # ego vehicle
            2: 255,  # rect border
            3: 255,  # out of roi
            4: 255,  # static
            5: 255,  # dynamic
            6: 255,  # ground
            7: 0,    # road
            8: 1,    # sidewalk
            9: 255,  # parking
            10: 255, # rail track
            11: 2,   # building
            12: 3,   # wall
            13: 4,   # fence
            14: 255, # guard rail
            15: 255, # bridge
            16: 255, # tunnel
            17: 5,   # pole
            18: 255, # polegroup
            19: 6,   # traffic light
            20: 7,   # traffic sign
            21: 8,   # vegetation
            22: 9,   # terrain
            23: 10,  # sky
            24: 11,  # person
            25: 12,  # rider
            26: 13,  # car
            27: 14,  # truck
            28: 15,  # bus
            29: 255, # caravan
            30: 255, # trailer
            31: 16,  # train
            32: 17,  # motorcycle
            33: 18,  # bicycle
            -1: 255  # ignore
        }
        
        # 添加CamVid标签映射
        self.camvid_classes = {
            (128, 128, 128): 0,    # Sky
            (128, 0, 0): 1,        # Building
            (192, 192, 128): 2,    # Pole
            (128, 64, 128): 3,     # Road
            (60, 40, 222): 4,      # Pavement
            (128, 128, 0): 5,      # Tree
            (192, 128, 128): 6,    # SignSymbol
            (64, 64, 128): 7,      # Fence
            (64, 0, 128): 8,       # Car
            (64, 64, 0): 9,        # Pedestrian
            (0, 128, 192): 10,     # Bicyclist
            (0, 0, 0): 255         # Void
        }
        
        # 根据数据集名称设置不同的目录结构
        if dataset_name == 'cityscapes':
            self.images_dir = os.path.join(data_dir, 'cityscapes', 'leftImg8bit', split)
            self.labels_dir = os.path.join(data_dir, 'cityscapes', 'gtFine', split)
            
            # 检查目录是否存在
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(
                    f"图像目录不存在: {self.images_dir}\n"
                    f"请确保数据集目录结构正确，当前目录结构应为:\n"
                    f"{data_dir}/\n"
                    f"├── cityscapes/\n"
                    f"│   ├── leftImg8bit/\n"
                    f"│   │   └── {split}/\n"
                    f"│   │       ├── aachen/\n"
                    f"│   │       └── ...\n"
                    f"│   └── gtFine/\n"
                    f"│       └── {split}/\n"
                    f"│           ├── aachen/\n"
                    f"│           └── ...\n"
                )
            
            if not os.path.exists(self.labels_dir):
                raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")
            
            # 获取所有子目录下的图像
            self.images = []
            self.labels = []
            city_count = 0
            for city in sorted(os.listdir(self.images_dir)):
                city_img_path = os.path.join(self.images_dir, city)
                city_label_path = os.path.join(self.labels_dir, city)
                
                if os.path.isdir(city_img_path):
                    city_count += 1
                    # 获取城市目录下的所有图像
                    city_images = [f for f in sorted(os.listdir(city_img_path)) 
                                 if f.endswith('leftImg8bit.png')]
                    
                    for img_name in city_images:
                        # 存储完整的相对路径
                        self.images.append(os.path.join(city, img_name))
                        self.labels.append(os.path.join(city, 
                            img_name.replace('leftImg8bit.png', 'gtFine_labelIds.png')))
            
            print(f"在 {split} 集中找到 {city_count} 个城市，共 {len(self.images)} 张图像")
            
            if len(self.images) == 0:
                raise RuntimeError("未找到任何图像文件，请检查数据集路径和文件格式")
                
        else:  # camvid
            self.images_dir = os.path.join(data_dir, 'camvid', split)
            self.labels_dir = os.path.join(data_dir, 'camvid', f'{split}_labels')
            
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(
                    f"图像目录不存在: {self.images_dir}\n"
                    f"请确保数据集目录结构正确，当前目录结构应为:\n"
                    f"{data_dir}/\n"
                    f"└── camvid/\n"
                    f"    ├── {split}/\n"
                    f"    └── {split}_labels/\n"
                )

            if not os.path.exists(self.labels_dir):
                raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")
            
            # 加载类别映射
            self.class_dict = pd.read_csv(os.path.join(data_dir, 'camvid', 'class_dict.csv'))
        
            # 获取图像列表
            self.images = [f for f in sorted(os.listdir(self.images_dir)) 
                          if f.endswith('.png')]
            self.labels = [f for f in sorted(os.listdir(self.labels_dir)) 
                          if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.dataset_name == 'cityscapes':
            # 使用完整的相对路径
            img_path = os.path.join(self.images_dir, self.images[idx])
            label_path = os.path.join(self.labels_dir, self.labels[idx])
            
            image = Image.open(img_path).convert('RGB')
            label = np.array(Image.open(label_path))
            
            # 将原始标签ID映射到训练ID
            label_copy = np.ones(label.shape, dtype=np.uint8) * 255
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            
            # 应用图像转换
            image = self.transform(image)
            
            # 处理标签
            label = Image.fromarray(label_copy)
            label = self.target_transform(label)  # 只进行resize
            label = torch.from_numpy(np.array(label))
            
            return image, label.long()
        else:  # camvid
            img_path = os.path.join(self.images_dir, self.images[idx])
            label_path = os.path.join(self.labels_dir, self.labels[idx])
        
            # 加载图像和标签
            image = Image.open(img_path).convert('RGB')
            label = np.array(Image.open(label_path))
            
            # 创建标签映射
            label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
            
            # 将RGB值映射到类别ID
            for rgb, class_id in self.camvid_classes.items():
                mask = np.all(label == rgb, axis=2)
                label_mask[mask] = class_id
            
            # 应用转换
            image = self.transform(image)
            
            # 处理标签
            label = Image.fromarray(label_mask)
            label = self.target_transform(label)
            label = torch.from_numpy(np.array(label)).long()
            
            return image, label

    def get_loader(self, batch_size, shuffle=True, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
