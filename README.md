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

启动一个包含教程的可视化网页界面：

```bash
python main.py
```

浏览器中将打开多个标签页，覆盖以下功能：

- **Fuzzy C-means**：二维数据的模糊聚类演示；
- **Density Peaks**：基于密度峰值的簇中心选择；
- **特征点检测**：上传人脸图片查看 2D 关键点；
- **UV 映射**：随机顶点生成并展示对应的 UV 坐标；
- **Delaunay**：对 UV 点集进行 Delaunay 三角剖分。

每个页面都包含算法简介与交互式示例，便于理解与调试。

## 模块简介

- **face_utils.ml**：密度聚类与模糊 C 均值实现。
- **face_utils.reconstruction**：人脸特征点检测、三角网格操作、渲染与拟合。
- **face_utils.registration**：提供 UV 映射、模型变换以及多种配准脚本示例。

## 许可证

本项目基于 MIT License 开源，欢迎学习与交流。
