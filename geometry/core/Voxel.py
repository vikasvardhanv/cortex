
# Render Utils
# Implements basic rasterization of SDFs into a Voxel Grid and marching cubes integration
import numpy as np
from geometry.core import SDF, BBox3, Mesh
import skimage.measure

class VoxelGrid:
    def __init__(self, size=None, resolution=0.1):
        """
        Initializes a dense voxel grid.
        :param size: Tuple (x, y, z) dimensions
        :param resolution: Voxel size in world units
        """
        self.resolution = resolution
        self.size = size if size else (100, 100, 100)
        self.grid = np.zeros(self.size, dtype=np.float32) + 1.0 # Default outside
        self.origin = np.array([0.0, 0.0, 0.0]) # World origin of the grid (0,0,0) index

    def from_sdf(self, sdf, padding_voxels=2):
        """
        Populate the grid from an SDF function.
        Automatically resizes the grid to fit the SDF bounds.
        """
        bounds = sdf.bounds()
        padding = padding_voxels * self.resolution
        min_pt = bounds.min_pt - padding
        max_pt = bounds.max_pt + padding
        size = (max_pt - min_pt) / self.resolution
        self.size = tuple(np.ceil(size).astype(int))
        self.origin = min_pt

        # Generate coordinate grid
        x = np.linspace(min_pt[0], max_pt[0], self.size[0])
        y = np.linspace(min_pt[1], max_pt[1], self.size[1])
        z = np.linspace(min_pt[2], max_pt[2], self.size[2])
        
        # Create meshgrid
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack((xv, yv, zv), axis=-1)
        
        # Calculate SDF
        # Flatten for batch processing if needed, but numpy handles broadcasting usually.
        # But our SDF.dist expects (N, 3).
        
        flat_points = points.reshape(-1, 3)
        
        # Batch processing to save memory
        batch_size = 100000
        distances = []
        for i in range(0, len(flat_points), batch_size):
            batch = flat_points[i:i+batch_size]
            d = sdf.dist(batch)
            distances.append(d)
        
        self.grid = np.concatenate(distances).reshape(self.size)
        print(f"Voxel Grid Created: {self.size} voxels from bounds {min_pt} to {max_pt}")

    def meshing(self):
        """
        Use Marching Cubes to generate a mesh from the voxel grid
        """
        # Marching cubes algorithm from skimage
        verts, faces, normals, values = skimage.measure.marching_cubes(self.grid, level=0.0)
        
        # Transform vertices back to world space
        # Verts are in grid coordinates (0 to size)
        # World = Grid * Resolution + Origin
        
        world_verts = verts * self.resolution + self.origin
        
        mesh = Mesh()
        mesh.vertices = world_verts
        mesh.faces = faces
        
        return mesh

