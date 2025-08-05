"""Gradio-based tutorial interface for the face_utils package."""

import numpy as np
import matplotlib.pyplot as plt

import gradio as gr
import cv2

from face_utils.ml import FCM, density_showing
from face_utils.reconstruction import landmark
from face_utils.registration import uvmap_processing, delaunay_processing


def demo_cluster() -> plt.Figure:
    """Run a tiny Fuzzy C-means clustering example and return a plot."""
    data = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9]], dtype=float)
    weight = np.ones(data.shape[1])
    model = FCM(n_clusters=2, random_state=0)
    model.fit(data, X_Weight=weight)
    labels = model.predict(data, X_Weight=weight)

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
    ax.scatter(model._centers[:, 0], model._centers[:, 1], marker="x", color="red", s=100)
    ax.set_title("Fuzzy C-means Clustering")
    return fig


def demo_density() -> plt.Figure:
    """Visualize the decision graph for density peaks clustering."""
    data = np.random.rand(30, 2)
    rho, delta = density_showing(data, flag=False)
    fig, ax = plt.subplots()
    ax.scatter(rho, delta, c="black")
    ax.set_xlabel("ρ (density)")
    ax.set_ylabel("δ (distance to higher density)")
    ax.set_title("Density Peaks Decision Graph")
    return fig


def demo_landmark(img: np.ndarray, method: str):
    """Detect facial landmarks on an image."""
    if img is None:
        return None, "请上传图片"
    try:
        pts = landmark.landmark2d_detect(img, method=method, flag_show=False)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for x, y in pts:
            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out, f"检测到 {len(pts)} 个特征点"
    except Exception as exc:  # pragma: no cover - interactive demo
        return None, f"检测失败: {exc}"


def demo_uvmap() -> plt.Figure:
    """Generate a UV map for a synthetic mesh and return a plot."""
    try:
        vertices = np.random.rand(100, 3)
        info = uvmap_processing.Vertices2Mapuv(vertices, flag_show=False, flag_select=False)
        uvmap = info.uvmap
        fig, ax = plt.subplots()
        ax.scatter(uvmap[:, 0], uvmap[:, 1], s=5)
        ax.set_title("UV Map")
        return fig
    except Exception as exc:  # pragma: no cover - interactive demo
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(exc), ha="center", va="center")
        ax.axis("off")
        return fig


def demo_delaunay() -> plt.Figure:
    """Triangulate UV points using Delaunay and show the mesh."""
    uv = np.random.rand(30, 2)
    try:
        triangles = delaunay_processing.mapuv_Delaunay(uv)
        fig, ax = plt.subplots()
        ax.triplot(uv[:, 0], uv[:, 1], triangles, color="gray")
        ax.scatter(uv[:, 0], uv[:, 1], color="red", s=10)
        ax.set_title("Delaunay Triangulation")
        return fig
    except Exception as exc:  # pragma: no cover - interactive demo
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(exc), ha="center", va="center")
        ax.axis("off")
        return fig


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Face Utils Web Demo\n交互式教程覆盖 `face_utils` 中的主要算法。")
        with gr.Tab("概览"):
            gr.Markdown(
                """
                ## 模块介绍
                - **ML**: 聚类算法，包括 Fuzzy C-means 与 Density Peaks。
                - **Reconstruction**: 面部特征点检测、3D 重建等工具。
                - **Registration**: UV 映射、Delaunay 三角剖分等配准相关方法。
                选择其他标签查看具体示例。
                """
            )
        with gr.Tab("Fuzzy C-means"):
            gr.Markdown("使用 Fuzzy C-means 对二维点集进行模糊聚类。")
            btn = gr.Button("运行示例")
            plot = gr.Plot()
            btn.click(demo_cluster, outputs=plot)
        with gr.Tab("Density Peaks"):
            gr.Markdown("展示密度峰值聚类的决策图，用于确定簇心。")
            btn_dp = gr.Button("运行示例")
            plot_dp = gr.Plot()
            btn_dp.click(demo_density, outputs=plot_dp)
        with gr.Tab("特征点检测"):
            gr.Markdown("上传人脸图像并选择检测器，查看二维特征点位置。")
            img_in = gr.Image(type="numpy", label="输入图像")
            method = gr.Radio(["dlib", "face_alignment", "baidu", "manual"], value="dlib", label="检测器")
            btn2 = gr.Button("开始检测")
            img_out = gr.Image(label="结果")
            info = gr.Textbox(label="信息")
            btn2.click(demo_landmark, inputs=[img_in, method], outputs=[img_out, info])
        with gr.Tab("UV 映射"):
            gr.Markdown("随机生成顶点并计算对应的 UV 坐标。")
            btn3 = gr.Button("生成示例")
            plot_uv = gr.Plot()
            btn3.click(demo_uvmap, outputs=plot_uv)
        with gr.Tab("Delaunay"):
            gr.Markdown("对 UV 点集进行 Delaunay 三角剖分。")
            btn4 = gr.Button("运行示例")
            plot_tri = gr.Plot()
            btn4.click(demo_delaunay, outputs=plot_tri)
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
