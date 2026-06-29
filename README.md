# DCGAN Hamburger Image Generation

基于 DCGAN（Deep Convolutional Generative Adversarial Network）的汉堡图像生成项目。

**作者**: Cradles

## 项目简介

本项目使用 MATLAB R2024a 实现 DCGAN，在 963 张手工裁剪的汉堡图像数据集（Burger_v2）上训练了一个 128×128 分辨率的汉堡图像生成模型。模型采用转置卷积生成器 + 卷积判别器的经典架构，结合 LSGAN 损失、Feature Matching 损失和 Minibatch StdDev 正则化提升训练稳定性和生成质量。

## 文件结构

```
food_gan_final/
├── code/
│   ├── project_of_food_final.m   # DCGAN 训练脚本（主程序）
│   ├── generate_burgers.m        # 加载模型直接生成 25 张图像
│   ├── generate_test.m           # 批量生成 + D 评分筛选 Top-25
│   ├── Burger_modify.m           # 数据集预处理（裁剪 + 白底填充至 256×256）
│   └── utils/
│       ├── minibatchStddevLayer.m  # 判别器 Minibatch StdDev 层
│       └── minibatchStddevFcn.m    # StdDev 核心函数
├── model/
│   └── hamburger_gan_final.mat   # 训练好的生成器/判别器权重（~84 MB）
├── results/
│   ├── final_burgers/            # 128×128 生成结果（25 张 + 5×5 拼贴图）
│   ├── final_burgers_512/        # 512×512 超分结果（25 张 + 拼贴图 + 训练曲线）
│   └── dataset_samples.png       # 数据集样本展示（25 张抽样）
└── README.md
```

## 环境要求

- **MATLAB R2024a** 或更高版本
- **必需工具箱**:
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - Parallel Computing Toolbox
  - Computer Vision Toolbox
- **推荐 GPU**: NVIDIA RTX 4060（8 GB 显存）或更高
- 训练显存需求约 5.5 GB

## 数据集

- **来源**: Fast Food Classification V2 数据集中的 Burger 类别
- **路径**: `D:\hamberger_datasets\Fast Food Classification V2\Train\Burger_v2`
- **规模**: 963 张 JPEG 图像
- **预处理**（`Burger_modify.m`）:
  1. 手工裁剪去除背景杂乱元素，仅保留汉堡主体
  2. 按较短边缩放，保持宽高比
  3. 居中放置于 256×256 纯白画布

## 使用方法

### 1. 训练模型

在 MATLAB 中运行 `code/project_of_food_final.m`：

```matlab
run('code/project_of_food_final.m');
```

- 选择 `[1]` 从零开始训练，或 `[2]` 从检查点续训
- 训练过程自动保存检查点至 `checkpoints_gan/` 目录
- 训练结束生成最终模型 `final_model.mat` 和最佳模型 `best_model.mat`

### 2. 生成图像（直接）

使用预训练模型直接生成 25 张图像（`code/generate_burgers.m`）：

```matlab
run('code/generate_burgers.m');
```

- 默认加载 `model/hamburger_gan_final.mat`
- 生成 25 张 128×128 汉堡图像
- 可选择保存路径

### 3. 生成图像（D 评分筛选）

批量生成 200 张候选图像，使用判别器自动评分，保留分数最高的 25 张（`code/generate_test.m`）：

```matlab
run('code/generate_test.m');
```

- 自动加载模型和工具函数
- 200 张候选 → 判别器 sigmoid 评分 → Top-25
- 输出单图 + 5×5 拼贴图 + 评分记录（`scores.txt`）

### 4. 数据预处理

如需重新预处理原始数据集：

```matlab
run('code/Burger_modify.m');
```

运行前请修改脚本中的 `inputFolder` 和 `outputFolder` 路径。

## 模型架构

### 生成器 (Generator)
- 输入: 128 维随机噪声向量
- 6 级转置卷积（kernel=4），通道数逐步减半（512→256→128→64→32→16→3）
- 每级含 BatchNorm + ReLU，输出层使用 tanh 激活
- 输出: 128×128×3 RGB 图像

### 判别器 (Discriminator)
- 输入: 128×128×3 图像
- 5 级卷积下采样（kernel=4, stride=2）
- 通道数: 64→128→256→256→256
- 含 Minibatch StdDev 层用于增强多样性感知
- 输出: 标量判别分数

### 损失函数
- **对抗损失**: LSGAN（Least Squares GAN）
- **特征匹配损失**: 单层特征匹配（λ=8），约束真实/生成图像在判别器中间层的特征一致性
- **正则化**: Minibatch StdDev + 梯度裁剪（D:50, G:25）

### 训练策略
- 优化器: Adam（D: β₁=0.0, G: β₁=0.5, β₂=0.999）
- 学习率: 余弦退火（G: 1e-4→5e-7, D: 3e-5→5e-7）
- 批量大小: 32
- EMA 生成器: β=0.999
- 自动崩溃检测与恢复

## 超分辨率后处理

生成图像（128×128）使用 **Real-ESRGAN** 进行 4× 超分辨率增强至 512×512。

- 项目地址: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- 使用版本: [realesrgan-ncnn-vulkan v0.2.0 (Windows)](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases)
- 工具路径: `D:\hamberger_datasets\realesrgan-ncnn-vulkan-20220424-windows.zip`

Real-ESRGAN 是一个开源盲超分辨率模型，由中国科学院深圳先进技术研究院的王鑫涛等人开发，在多种真实场景退化图像上表现出色。

## 实验结果

- 生成图像成功保留了汉堡的多层堆叠结构（面包→配料→肉饼→面包）
- 不同样本在面包色泽、配料排列和构图上具有明显多样性
- 训练过程稳定，未发生模式崩溃
- FID 指标随训练持续下降，验证了生成质量的改善

## 参考文献

1. Goodfellow et al. (2014) — Generative Adversarial Nets
2. Radford et al. (2016) — DCGAN
3. Mao et al. (2017) — LSGAN
4. Salimans et al. (2016) — Improved Techniques for Training GANs
5. Heusel et al. (2017) — GANs Trained by a Two Time-Scale Update Rule (FID)
6. Karras et al. (2018) — Progressive Growing of GANs (Minibatch StdDev)
7. Wang et al. (2018) — ESRGAN
8. Wang et al. (2021) — Real-ESRGAN: Training Real-World Blind Super-Resolution
9. xinntao (2022) — Real-ESRGAN-ncnn-vulkan (超分辨率工具)
10. Saxena U. — Fast Food Classification Dataset V2 (Kaggle 数据集)
