# Face Utils

该项目提供了一组用于三维人脸处理的工具库, 并经过重构使其模块化和清晰化。主要包含三个子模块:

- **face_utils.ml**: 聚类与模糊 C 均值等机器学习算法。
- **face_utils.reconstruction**: 三维人脸重建相关的工具, 包括特征点检测、网格操作、渲染与拟合等。
- **face_utils.registration**: 人脸配准与对齐流程, 提供 UV 映射、模型变换以及多种配准脚本。

## 安装

```bash
pip install -e .
```

## 使用示例

```python
from face_utils.reconstruction import landmark, mesh

# 加载并检测特征点
points = landmark.landmark2d_detect(image)
```

更多示例可参考 `face_utils/registration/face_registration` 目录下的脚本。

## 目录结构

```
face_utils/
  ml/                 # 聚类与模糊 C 均值
  reconstruction/     # 重建相关功能
  registration/       # 配准与对齐
```

## 许可证

本项目遵循 MIT License。
