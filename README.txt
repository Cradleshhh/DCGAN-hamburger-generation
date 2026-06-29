============================================================
  DCGAN 汉堡图像生成 — 课程项目
  Cradles
============================================================

本项目完整代码、生成结果和课程论文已上传至 GitHub：

    https://github.com/[你的用户名]/dcgan-hamburger-generation

（请将上述链接替换为实际的 GitHub 仓库地址）

项目包含以下内容：

1. code/ — MATLAB 源代码
   - project_of_food_final.m : DCGAN 训练主程序
   - generate_burgers.m      : 模型加载 + 图像生成
   - Burger_modify.m         : 数据集预处理

2. model/ — 预训练模型权重
   - hamburger_gan_final.mat (84 MB)

3. results/ — 生成结果
   - final_burgers/          : 128×128 原始生成结果
   - final_burgers_512/      : 512×512 超分辨率结果
   - dataset_samples.png     : 数据集样本展示

4. report.pdf — 课程论文/实验报告（LaTeX 源文件 report.tex 亦已上传）

5. README.md — 项目说明文档（含环境配置、使用方法、架构说明）

训练环境：
  - MATLAB R2024a + Deep Learning Toolbox
  - NVIDIA RTX 4060 8GB GPU
  - 数据集: Burger_v2 (963 张手工裁剪汉堡图像)

超分辨率后处理：
  - 使用开源工具 Real-ESRGAN (xinntao/Real-ESRGAN-ncnn-vulkan)
  - 将 128×128 生成结果增强至 512×512

如有任何疑问，请在 GitHub 仓库提交 Issue。
