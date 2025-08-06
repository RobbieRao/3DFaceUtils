# 3DFaceUtils 教程项目

本仓库由 Robbie 在 2025 年编写整理，涵盖聚类、重建以及配准三个部分，并提供一个多合一的演示脚本 `main.py`。

## 安装步骤

1. 准备 Python 3.8 及以上环境。
2. 克隆仓库并安装依赖：
   ```bash
   pip install -e .
   ```
3. 部分功能依赖额外库（如 `dlib`、`open3d`、`mayavi` 等），按需安装即可。

## 数据准备

- **聚类与示例数据**：`main.py` 内置了随机生成的点集，可直接运行示例；若需在自定义数据上实验，准备形如 `N x D` 的 `numpy` 数组即可。
- **特征点检测**：需要输入 RGB 人脸图像（`.jpg`、`.png` 等）。使用 `dlib` 检测器时还需提前下载 `shape_predictor_68_face_landmarks.dat` 等模型文件。
- **UV 映射与配准**：示例中随机生成顶点；真实应用通常需要三维网格文件（如 `.obj`、`.ply`）及其对应的 UV 坐标或人脸对应关系数据。

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

## 常用函数示例

```python
from face_utils.ml import FCM
model = FCM(n_clusters=3)
model.fit(data, X_Weight=weight)
labels = model.predict(data, X_Weight=weight)

from face_utils.reconstruction import landmark
pts = landmark.landmark2d_detect(img, method="dlib")

from face_utils.registration import uvmap_processing, delaunay_processing
uv = uvmap_processing.Vertices2Mapuv(vertices).uvmap
tri = delaunay_processing.mapuv_Delaunay(uv)
```

- `FCM.fit` / `predict`：模糊 C 均值聚类。
- `density_showing`：计算并可视化密度峰值聚类的决策图。
- `landmark.landmark2d_detect`：在图像上检测 2D 特征点。
- `uvmap_processing.Vertices2Mapuv`：为顶点生成 UV 坐标。
- `delaunay_processing.mapuv_Delaunay`：对 UV 点集进行 Delaunay 三角剖分。

## 模块简介

- **face_utils.ml**：密度聚类与模糊 C 均值实现。
- **face_utils.reconstruction**：人脸特征点检测、三角网格操作、渲染与拟合。
- **face_utils.registration**：提供 UV 映射、模型变换以及多种配准脚本示例。

## 项目用途

该项目用于快速体验和验证三维人脸处理相关的 **聚类、重建与配准** 算法，适合教学演示、课程实验以及算法原型开发。

## 许可证

本项目基于 MIT License 开源，欢迎学习与交流。
