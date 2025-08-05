# 3DFaceUtils 教程项目

本仓库由 Robbie 在 2025 年整理，旨在帮助刚入门的同学快速体验三维人脸处理相关工具。项目按照教程形式组织，涵盖聚类、重建以及配准三个部分，并提供一个多合一的演示脚本 `main.py`。

## 安装步骤

1. 准备 Python 3.8 及以上环境。
2. 克隆仓库并安装依赖：
   ```bash
   pip install -e .
   ```
3. 部分功能依赖额外库（如 `dlib`、`open3d`、`mayavi` 等），按需安装即可。

## 目录结构

```text
face_utils/
  ml/               # 聚类算法与模糊 C 均值
  reconstruction/   # 特征点检测、网格处理与可视化
  registration/     # UV 映射与模型配准
main.py             # 多合一演示脚本
```

## 快速体验

### 1. 聚类演示
```bash
python main.py cluster
```
脚本会生成一些示例点并使用模糊 C 均值进行聚类。

### 2. 图像特征点检测
```bash
python main.py landmark path/to/image.jpg
```
默认使用 dlib 作为检测器，可通过 `--method` 参数选择 `face_alignment`、`baidu` 或 `manual`。

### 3. UV Map 生成
```bash
python main.py uvmap
```
随机生成一个网格并计算其 UV 映射，适合理解配准相关流程。

## 模块简介

- **face_utils.ml**：密度聚类与模糊 C 均值实现。
- **face_utils.reconstruction**：人脸特征点检测、三角网格操作、渲染与拟合。
- **face_utils.registration**：提供 UV 映射、模型变换以及多种配准脚本示例。

## 许可证

本项目基于 MIT License 开源，欢迎学习与交流。
