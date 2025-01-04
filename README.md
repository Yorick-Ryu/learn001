# 二维图像的语义分割网络

- [二维图像的语义分割网络](#二维图像的语义分割网络)
  - [1. 环境](#1-环境)
    - [1.1. 基本环境](#11-基本环境)
    - [1.2. 环境管理工具](#12-环境管理工具)
    - [1.3. 具体环境](#13-具体环境)
  - [2. 任务要求](#2-任务要求)
  - [3. 任务实现](#3-任务实现)
    - [3.1. 实现内容](#31-实现内容)
    - [3.2. 使用方法：](#32-使用方法)
    - [3.3. 数据集](#33-数据集)
      - [3.3.1. 数据集目录结构](#331-数据集目录结构)
      - [3.3.2. 数据集说明](#332-数据集说明)
      - [3.3.3. 数据预处理](#333-数据预处理)
    - [3.4. 网络架构](#34-网络架构)
      - [3.4.1. UNet](#341-unet)
      - [3.4.2. ResNet-Based](#342-resnet-based)
      - [3.4.3. 网络对比](#343-网络对比)
    - [3.5. 训练过程可视化](#35-训练过程可视化)
      - [3.5.1. TensorBoard使用方法](#351-tensorboard使用方法)
      - [3.5.2. 可视化内容](#352-可视化内容)
      - [3.5.3. 多次训练结果对比](#353-多次训练结果对比)
  - [4. 后续计划](#4-后续计划)
  - [5. 版本控制说明](#5-版本控制说明)
    - [5.1. Git管理](#51-git管理)
    - [5.2. 数据集获取](#52-数据集获取)


## 1. 环境

### 1.1. 基本环境

- Python 3.10.15
- PyTorch 版本: 2.5.0+cu124
- CUDA 版本: 12.4
- CUDA 设备名称: NVIDIA GeForce GTX 1650

### 1.2. 环境管理工具
- miniconda
- 创建pytorch环境并激活

### 1.3. 具体环境
<details>
<summary>点击展开查看详细环境信息</summary>

```
Collecting environment information...
PyTorch version: 2.5.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.6.77
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce GTX 1650
Nvidia driver version: 560.94
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      48 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             12
On-line CPU(s) list:                0-11
Vendor ID:                          AuthenticAMD
Model name:                         AMD Ryzen 5 4600H with Radeon Graphics
CPU family:                         23
Model:                              96
Thread(s) per core:                 2
Core(s) per socket:                 6
Socket(s):                          1
Stepping:                           1
BogoMIPS:                           5988.55
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip rdpid
Virtualization:                     AMD-V
Hypervisor vendor:                  Microsoft
Virtualization type:                full
L1d cache:                          192 KiB (6 instances)
L1i cache:                          192 KiB (6 instances)
L2 cache:                           3 MiB (6 instances)
L3 cache:                           4 MiB (1 instance)
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Mitigation; untrained return thunk; SMT enabled with STIBP protection
Vulnerability Spec rstack overflow: Mitigation; safe RET
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Retpolines, IBPB conditional, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected

Versions of relevant libraries:
[pip3] numpy==1.23.5
[pip3] torch==2.5.0
[pip3] torch-tb-profiler==0.4.3
[pip3] torchvision==0.20.0
[pip3] triton==3.1.0
[conda] numpy                     1.23.5                   pypi_0    pypi
[conda] torch                     2.5.0                    pypi_0    pypi
[conda] torch-tb-profiler         0.4.3                    pypi_0    pypi
[conda] torchvision               0.20.0                   pypi_0    pypi
[conda] triton                    3.1.0                    pypi_0    pypi
```
</details>

## 2. 任务要求

使用tensorflow/pytorch框架，配置一个二维图像的语义分割网络。要求5号之前完成：
1. 能够根据参数选择加载UNet/Resnet网络；
2. 能够根据参数选择加载不同的语义分割数据集（cityscapes、CamVid）；
3. 能够加载一个预训练模型；
4. 能够使用tensorboard或torchvision对训练过程可视化。

## 3. 任务实现

### 3.1. 实现内容

1. 基础训练脚本框架搭建
   - 参数解析设置
   - 数据加载器接口
   - 模型选择功能（UNet/ResNet）
   - TensorBoard可视化支持

2. 训练循环实现
   - GPU训练支持
   - 损失计算
   - 优化器设置
   - 进度显示
   - 模型保存

**关键特性**

- 支持多种模型架构选择
- 支持预训练模型加载
- 支持不同数据集（cityscapes/camvid）
- 自动GPU/CPU设备选择
- TensorBoard集成
- 定期模型检查点保存

### 3.2. 使用方法：

```bash
python train.py --model [unet/resnet] \
                --dataset [cityscapes/camvid] \
                --data_dir DATA_DIR \
                --log_dir LOG_DIR \
                --batch_size BATCH_SIZE \
                --epochs EPOCHS \
                --num_classes NUM_CLASSES \
                [--pretrained PRETRAINED_MODEL_PATH]
```

具体运行命令示例

示例1：
```bash
python train.py --model unet --dataset cityscapes --data_dir ./data --log_dir ./logs --batch_size 2 --epochs 50 --num_classes 19
```
示例2：
```bash
python train.py --model resnet \
                --dataset camvid \
                --data_dir ./data \
                --log_dir ./logs/camvid/resnet \
                --batch_size 16 \
                --epochs 50 \
                --num_classes 11 
```

### 3.3. 数据集

#### 3.3.1. 数据集目录结构

项目支持两种数据集：Cityscapes 和 CamVid，它们的目录结构如下：

```
cityscapes
│
├── leftImg8bit/     # 原始图像
│   ├── train/
│   ├── val/
│   └── test/
│
└── gtFine/          # 精细标注
    ├── train/
    ├── val/
    └── test/
```

```
camvid
│
├── train/           # 训练图像
├── train_labels/    # 训练标签
├── val/             # 验证图像
├── val_labels/      # 验证标签
├── test/            # 测试图像
└── test_labels/     # 测试标签
```

#### 3.3.2. 数据集说明

1. Cityscapes
   - 图像格式：PNG
   - 训练图像尺寸：2048x1024
   - 类别数：19
   - 下载地址：https://www.cityscapes-dataset.com/
   
2. CamVid
   - 图像格式：PNG
   - 训练图像尺寸：960x720
   - 类别数：11
   - 下载地址：http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

#### 3.3.3. 数据预处理
- 所有图像会被调整为统一大小：512x512
- 图像值范围：0-1（归一化）
- 标签编码：整数编码（0 到 num_classes-1）

### 3.4. 网络架构

#### 3.4.1. UNet
UNet是一种经典的编码器-解码器结构的分割网络：

- 结构特点：
  - U形对称结构
  - 包含4次下采样和4次上采样
  - 特征图通过跳跃连接传递
  - 使用双卷积块进行特征提取

- 优点：
  - 结构简单，易于理解和实现
  - 适用于小规模数据集
  - 跳跃连接有助于保留细节信息
  - 训练速度较快

- 局限性：
  - 感受野相对较小
  - 缺乏全局上下文信息
  - 特征提取能力相对有限

#### 3.4.2. ResNet-Based
基于ResNet的分割网络采用预训练的ResNet作为骨干网络：

- 结构特点：
  - 使用预训练的ResNet-50作为编码器
  - 残差连接帮助训练更深的网络
  - 通过转置卷积进行上采样
  - 强大的特征提取能力

- 优点：
  - 可以利用预训练模型
  - 更强的特征提取能力
  - 适合处理复杂场景
  - 有更好的泛化能力

- 局限性：
  - 模型参数量大
  - 训练和推理速度较慢
  - 需要更多的计算资源
  - 可能过度关注高层语义特征

#### 3.4.3. 网络对比

| 特性 | UNet | ResNet-Based |
|------|------|--------------|
| 参数量 | 较少（约7M） | 较多（约25M） |
| 训练速度 | 快 | 慢 |
| 推理速度 | 快 | 慢 |
| 预训练支持 | 无 | 有 |
| 特征提取能力 | 中等 | 强 |
| 内存占用 | 低 | 高 |
| 适用场景 | 医学图像等简单场景 | 自动驾驶等复杂场景 |


### 3.5. 训练过程可视化

#### 3.5.1. TensorBoard使用方法

1. 启动TensorBoard服务：
```bash
tensorboard --logdir=logs --port=6006
```

2. 在浏览器中访问：http://localhost:6006

#### 3.5.2. 可视化内容

1. SCALARS标签页：
   - 训练损失曲线（Loss/train）
   - 学习率变化（Learning_rate）
   
2. IMAGES标签页：
   - 输入图像（Image/input）
   - 目标分割图（Image/target）
   - 预测分割图（Image/pred）
   
3. HISTOGRAMS标签页：
   - 模型参数分布（weight/*）
   - 梯度分布（grad/*）

#### 3.5.3. 多次训练结果对比

可以同时查看多个实验的结果：

```bash
# 查看所有实验结果
tensorboard --logdir=logs

# 查看特定实验结果
tensorboard --logdir=logs/cityscapes/resnet  # ResNet在Cityscapes上的结果
tensorboard --logdir=logs/camvid/unet        # UNet在CamVid上的结果
```

## 4. 后续计划

1. 添加验证集评估
2. 实现多个评估指标
3. 添加数据增强
4. 支持多GPU训练
5. 添加学习率调度器

## 5. 版本控制说明

### 5.1. Git管理

项目使用Git进行版本控制。以下文件和目录已被添加到.gitignore：

- `data/`：数据集目录
- `cityscapes/`：Cityscapes数据集
- `camvid/`：CamVid数据集
- `logs/`：训练日志
- `checkpoints/`：模型检查点
- `*.pth`, `*.pt`, `*.ckpt`：模型文件
- Python相关：`__pycache__/`, `*.pyc`等
- IDE相关：`.idea/`, `.vscode/`等

### 5.2. 数据集获取

由于数据集文件较大，不包含在版本控制中。请按以下步骤获取数据集：

1. Cityscapes数据集：
   - 访问 https://www.cityscapes-dataset.com/
   - 注册并下载数据集
   - 将数据集解压到 `data/cityscapes/` 目录

2. CamVid数据集：
   - 访问 http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
   - 下载数据集
   - 将数据集解压到 `data/camvid/` 目录
