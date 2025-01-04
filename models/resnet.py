import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetSegmentation(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(ResNetSegmentation, self).__init__()
        
        # 使用新的weights参数替代pretrained
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            # 2048 -> 1024, 8x8 -> 16x16
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # 1024 -> 512, 16x16 -> 32x32
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 512 -> 256, 32x32 -> 64x64
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 -> 128, 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 -> n_classes, 128x128 -> 256x256
            nn.ConvTranspose2d(128, n_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 添加尺寸检查
        input_size = x.size()[-2:]
        x = self.encoder(x)
        x = self.decoder(x)
        
        # 确保输出尺寸正确
        if x.size()[-2:] != input_size:
            print(f"Warning: Output size {x.size()[-2:]} doesn't match input size {input_size}")
        return x
