
# Shapes Implementation for Cortex Geometry Engine
# Implements basic SDFs for common shapes

from geometry.core import SDF, BBox3
import numpy as np

class Sphere(SDF):
    def __init__(self, radius=1.0, center=[0.0, 0.0, 0.0]):
        self.radius = radius
        self.center = np.array(center)
    
    def dist(self, p):
        # p is a numpy array of shape (N, 3)
        # return dist(p, center) - radius
        return np.linalg.norm(p - self.center, axis=-1) - self.radius
    
    def bounds(self):
        b = BBox3()
        b.min_pt = self.center - self.radius
        b.max_pt = self.center + self.radius
        return b

class Box(SDF):
    def __init__(self, size=[1.0, 1.0, 1.0], center=[0.0, 0.0, 0.0]):
        self.size = np.array(size)
        self.center = np.array(center)
    
    def dist(self, p):
        # p is a numpy array (N, 3)
        # d = abs(p - center) - size / 2.0
        d = np.abs(p - self.center) - self.size / 2.0
        return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(np.max(d, axis=-1), 0.0)

    def bounds(self):
        b = BBox3()
        b.min_pt = self.center - self.size / 2.0
        b.max_pt = self.center + self.size / 2.0
        return b

class Cylinder(SDF):
    def __init__(self, radius=1.0, height=2.0, center=[0.0, 0.0, 0.0]):
        self.radius = radius
        self.height = height
        self.center = np.array(center)
    
    def dist(self, p):
        # p is a numpy array (N, 3)
        local_p = p - self.center
        # Assuming cylinder aligned to Z-axis
        # dist to axis
        d_axis = np.linalg.norm(local_p[..., :2], axis=-1) - self.radius
        d_height = np.abs(local_p[..., 2]) - self.height / 2.0
        return np.maximum(d_axis, d_height)
    
    def bounds(self):
        b = BBox3()
        b.min_pt = self.center - np.array([self.radius, self.radius, self.height/2.0])
        b.max_pt = self.center + np.array([self.radius, self.radius, self.height/2.0])
        return b

class Torus(SDF):
    def __init__(self, major_radius=2.0, minor_radius=0.5, center=[0.0, 0.0, 0.0]):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.center = np.array(center)
    
    def dist(self, p):
        # p is a numpy array (N, 3)
        local_p = p - self.center
        # Assuming horizontal torus in XY plane
        q = np.array([np.linalg.norm(local_p[..., :2], axis=-1) - self.major_radius, local_p[..., 2]])
        return np.linalg.norm(q, axis=0) - self.minor_radius
    
    def bounds(self):
        b = BBox3()
        total_radius = self.major_radius + self.minor_radius
        b.min_pt = self.center - np.array([total_radius, total_radius, self.minor_radius])
        b.max_pt = self.center + np.array([total_radius, total_radius, self.minor_radius])
        return b
