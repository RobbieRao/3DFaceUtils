import numpy as np
from scipy.spatial import cKDTree


class TriMesh:
    """Simple triangle mesh placeholder built on numpy."""

    def __init__(self, points=None, trilist=None):
        self.points = np.array(points) if points is not None else np.zeros((0, 3))
        self.trilist = np.array(trilist) if trilist is not None else np.zeros((0, 3), dtype=int)
        self.landmarks = {}

    def copy(self):
        return TriMesh(self.points.copy(), self.trilist.copy())

    def centre(self):
        return self.points.mean(axis=0) if len(self.points) else np.zeros(3)

    def range(self):
        return self.points.ptp(axis=0) if len(self.points) else np.zeros(3)


class PointCloud:
    """Placeholder for a point cloud."""

    def __init__(self, points=None):
        self.points = np.array(points) if points is not None else np.zeros((0, 3))


class _BaseTransform:
    def apply(self, obj):
        return obj

    def compose_before(self, other):
        return CompositeTransform(self, other)

    def pseudoinverse(self):
        return self


class CompositeTransform(_BaseTransform):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def apply(self, obj):
        return self.first.apply(self.second.apply(obj))

    def pseudoinverse(self):
        return CompositeTransform(self.second.pseudoinverse(), self.first.pseudoinverse())


class Translation(_BaseTransform):
    def __init__(self, vector):
        self.vector = np.array(vector)

    def apply(self, obj):
        if hasattr(obj, "points"):
            obj.points = obj.points + self.vector
            return obj
        return obj + self.vector

    def pseudoinverse(self):
        return Translation(-self.vector)


class UniformScale(_BaseTransform):
    def __init__(self, scale, n_dims=None):
        self.scale = scale

    def apply(self, obj):
        if hasattr(obj, "points"):
            obj.points = obj.points * self.scale
            return obj
        return obj * self.scale

    def pseudoinverse(self):
        return UniformScale(1.0 / self.scale)


class AlignmentSimilarity(_BaseTransform):
    def __init__(self, *args, **kwargs):
        pass

    def as_non_alignment(self):
        return self


def trimesh_to_vtk(mesh):
    return mesh


class VTKClosestPointLocator:
    def __init__(self, mesh):
        self.points = np.asarray(mesh.points)
        self.kd = cKDTree(self.points)

    def __call__(self, query_points):
        _, indices = self.kd.query(query_points)
        return self.points[indices], indices


class ShapeModel:
    pass


def non_rigid_icp(*args, **kwargs):
    return None


class Mesh(TriMesh):
    pass


def read_mesh(path):
    """Placeholder mesh reader returning an empty mesh."""
    return Mesh()
