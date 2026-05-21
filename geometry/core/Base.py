
# MyCEM
# A Python-based Computational Engineering Model inspired by PicoGK
# Built for creating complex engineering geometries (like Rocket Engines)

import numpy as np

class Vector3:
    """Simple specific Vector 3 type for clarity, though we mostly use numpy arrays"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def to_np(self):
        return np.array([self.x, self.y, self.z])

class BBox3:
    """Axis Aligned Bounding Box"""
    def __init__(self, min_pt=None, max_pt=None):
        self.min_pt = np.array(min_pt) if min_pt is not None else np.array([float('inf')]*3)
        self.max_pt = np.array(max_pt) if max_pt is not None else np.array([float('-inf')]*3)

    def expand(self, point):
        self.min_pt = np.minimum(self.min_pt, point)
        self.max_pt = np.maximum(self.max_pt, point)

    def center(self):
        return (self.min_pt + self.max_pt) * 0.5
    
    def size(self):
        return self.max_pt - self.min_pt

class Mesh:
    """Simple triangular mesh structure"""
    def __init__(self):
        self.vertices = [] # List of [x, y, z]
        self.faces = []    # List of [v1, v2, v3] indices
    
    def add_vertex(self, x, y, z):
        self.vertices.append([x, y, z])
        return len(self.vertices) - 1
    
    def add_face(self, v1, v2, v3):
        self.faces.append([v1, v2, v3])

    def save_obj(self, filename):
        with open(filename, 'w') as f:
            f.write("# MyCEM Mesh\n")
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in self.faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Saved mesh to {filename} with {len(self.vertices)} vertices and {len(self.faces)} faces")

class SDF:
    """Abstract Base Class for Signed Distance Functions"""
    def dist(self, p):
        """
        Returns the signed distance from point p to the surface.
        p is a numpy array of shape (N, 3) for vectorized operations.
        Returns array of shape (N,)
        """
        raise NotImplementedError
    
    def bounds(self):
        """Returns BBox3 of the object"""
        raise NotImplementedError

# Core CSG Operations on SDFs
class Union(SDF):
    def __init__(self, sdf1, sdf2):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
    
    def dist(self, p):
        return np.minimum(self.sdf1.dist(p), self.sdf2.dist(p))
    
    def bounds(self):
        b = BBox3()
        b1 = self.sdf1.bounds()
        b2 = self.sdf2.bounds()
        b.min_pt = np.minimum(b1.min_pt, b2.min_pt)
        b.max_pt = np.maximum(b1.max_pt, b2.max_pt)
        return b

class Difference(SDF):
    def __init__(self, sdf1, sdf2):
        self.sdf1 = sdf1 # Main object
        self.sdf2 = sdf2 # Object to subtract
    
    def dist(self, p):
        return np.maximum(self.sdf1.dist(p), -self.sdf2.dist(p))

    def bounds(self):
        return self.sdf1.bounds() # Conservative bounds (ignoring subtraction)

class Intersection(SDF):
    def __init__(self, sdf1, sdf2):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
    
    def dist(self, p):
        return np.maximum(self.sdf1.dist(p), self.sdf2.dist(p))
    
    def bounds(self):
        b = BBox3()
        b1 = self.sdf1.bounds()
        b2 = self.sdf2.bounds()
        b.min_pt = np.maximum(b1.min_pt, b2.min_pt) # Intersection is smaller
        b.max_pt = np.minimum(b1.max_pt, b2.max_pt)
        return b

class Offset(SDF):
    def __init__(self, sdf, offset):
        self.sdf = sdf
        self.offset = offset # Positive expands, Negative contracts
    
    def dist(self, p):
        return self.sdf.dist(p) - self.offset

    def bounds(self):
        b = self.sdf.bounds()
        b.min_pt -= self.offset
        b.max_pt += self.offset
        return b

class SmoothUnion(SDF):
    def __init__(self, sdf1, sdf2, k):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.k = k
    
    def dist(self, p):
        d1 = self.sdf1.dist(p)
        d2 = self.sdf2.dist(p)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.k, 0.0, 1.0)
        return self.mix(d2, d1, h) - self.k * h * (1.0 - h)
    
    def mix(self, x, y, a):
        return x * (1 - a) + y * a

    def bounds(self):
        return Union(self.sdf1, self.sdf2).bounds() # Approximate
