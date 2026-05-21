
# CSG Operations and Utility Classes
from geometry.core import SDF
import numpy as np

class Translate(SDF):
    def __init__(self, sdf, offset):
        self.sdf = sdf
        self.offset = np.array(offset) # Vector (3,)
    
    def dist(self, p):
        return self.sdf.dist(p - self.offset)

    def bounds(self):
        b = self.sdf.bounds()
        b.min_pt += self.offset
        b.max_pt += self.offset
        return b

class RotateMC(SDF):
    # Marching Cubes SDF rotation is trickier without matrix lib, so using simple axis rotation for now
    def __init__(self, sdf, axis='z', angle_deg=0):
        self.sdf = sdf
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        self.axis = axis
        self.m = np.eye(3)
        if axis == 'z':
            self.m = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        elif axis == 'y':
            self.m = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis == 'x':
            self.m = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
            
    def dist(self, p):
        # Inverse transform the point
        # R * p_inverse = p => p_inverse = R^T * p
        p_rot = np.dot(p, self.m)
        return self.sdf.dist(p_rot)
    
    def bounds(self):
        # Naive bounds transform
        # Transform all 8 corners
        b = self.sdf.bounds()
        corners = []
        for x in [b.min_pt[0], b.max_pt[0]]:
            for y in [b.min_pt[1], b.max_pt[1]]:
                for z in [b.min_pt[2], b.max_pt[2]]:
                    corners.append([x, y, z])
        corners = np.array(corners)
        corners_rot = np.dot(corners, self.m.T) # Rotate forward
        
        # New bbox
        new_min = np.min(corners_rot, axis=0)
        new_max = np.max(corners_rot, axis=0)
        
        from geometry.core import BBox3
        return BBox3(new_min, new_max)
