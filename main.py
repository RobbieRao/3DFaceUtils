"""Multi-tool demonstration script for the face_utils package.

This tutorial-style script exposes several small demos so beginners can
quickly try out clustering, landmark detection and UV map generation.
"""
import argparse
import numpy as np

def demo_cluster(_: argparse.Namespace) -> None:
    """Run a tiny Fuzzy C-means clustering example."""
    from face_utils.ml import FCM

    data = np.array([[0,0], [0,1], [1,0], [9,9], [9,8], [8,9]], dtype=float)
    weight = np.ones(data.shape[1])
    model = FCM(n_clusters=2, random_state=0)
    model.fit(data, X_Weight=weight)
    labels = model.predict(data, X_Weight=weight)
    print("Cluster centers:\n", model._centers)
    print("Labels:", labels)

def demo_landmark(args: argparse.Namespace) -> None:
    """Detect facial landmarks on an image."""
    try:
        import cv2
        from face_utils.reconstruction import landmark
    except Exception as exc:
        print("Landmark demo requires OpenCV and dlib:", exc)
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read image: {args.image}")
        return
    pts = landmark.landmark2d_detect(img, method=args.method, flag_show=False)
    print(f"Detected {len(pts)} landmark points")

def demo_uvmap(_: argparse.Namespace) -> None:
    """Generate a UV map for a synthetic mesh."""
    try:
        from face_utils.registration import uvmap_processing
    except Exception as exc:
        print("UV map demo requires additional visualization libraries:", exc)
        return

    vertices = np.random.rand(100, 3)
    uvmap = uvmap_processing.Vertices2Mapuv(vertices, flag_show=False, flag_select=False)
    print("Generated UV map with shape", uvmap.shape)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Utils tutorial demo")
    sub = parser.add_subparsers(dest="command")

    p_cluster = sub.add_parser("cluster", help="Run Fuzzy C-means clustering demo")
    p_cluster.set_defaults(func=demo_cluster)

    p_landmark = sub.add_parser("landmark", help="Detect landmarks on an image")
    p_landmark.add_argument("image", help="Path to an input image")
    p_landmark.add_argument("--method", default="dlib", choices=["dlib", "face_alignment", "baidu", "manual"], help="Detection backend")
    p_landmark.set_defaults(func=demo_landmark)

    p_uvmap = sub.add_parser("uvmap", help="Generate a UV map for a random mesh")
    p_uvmap.set_defaults(func=demo_uvmap)

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
