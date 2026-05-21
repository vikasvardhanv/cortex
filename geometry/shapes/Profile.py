
# 2D Shapes extruding profile
# Useful for Revolve

class Shape2D:
    pass

class Circle2D(Shape2D):
    # Returns 2D SDF
    def dist(self, p):
        # p is (N, 2)
        return np.linalg.norm(p) - 1.0

# Revolve Operation
from geometry.core import SDF
import numpy as np

class Revolve(SDF):
    """Revolves a 2D SDF around an axis to create a 3D solid"""
    def __init__(self, sdf2d, axis='z'):
        self.sdf2d = sdf2d
        self.axis = axis # Currently supports Z axis revolution from XZ plane
    
    def dist(self, p):
        # p is (N, 3)
        # Convert (x, y, z) to (r, z)
        # r = sqrt(x^2 + y^2)
        r = np.linalg.norm(p[:, :2], axis=1)
        z = p[:, 2]
        
        q = np.stack([r, z], axis=1) # (N, 2)
        return self.sdf2d.dist(q)

