import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
import trimesh

# 2.4.1 Clustering with Voxel Representation – helpers and demo
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


def get_modelnet10_metadata():
    # read the metadata_modelnet10.csv file
    meta_data_path = os.path.join(os.path.dirname(__file__), "metadata_modelnet10.csv")
    metadata = pd.read_csv(meta_data_path)
    return metadata


# Reading .off file manually
def read_off(shape_filename):
    with open(shape_filename, "r") as file:
        if "OFF" != file.readline().strip():
            raise ("Not a valid OFF header")
        n_verts, n_faces, __ = tuple(
            [int(s) for s in file.readline().strip().split(" ")]
        )
        verts = [
            [float(s) for s in file.readline().strip().split(" ")]
            for _ in range(n_verts)
        ]
        faces = [
            [int(s) for s in file.readline().strip().split(" ")][1:]
            for _ in range(n_faces)
        ]
        file.close()

        return np.array(verts), np.array(faces)


def calculate_vertex_normals_trimesh(vertices, faces):
    """
    Calculate vertex normals using trimesh

    Args:
        vertices: numpy array of shape (n_vertices, 3)
        faces: numpy array of shape (n_faces, 3) - triangular faces

    Returns:
        vertex_normals: numpy array of shape (n_vertices, 3)
    """
    # Create trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    # Get vertex normals
    vertex_normals = mesh.vertex_normals

    return vertex_normals


# Utilities
def load_mesh_from_relpath(full_path):
    verts, faces = read_off(full_path)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def voxelize(mesh, grid_size):
    """
    Takes a trimesh and converts it into a voxel grid of size grid_size x grid_size x grid_size

    Args:
        mesh: trimesh.Trimesh object
        grid_size: int, size of the voxel grid (e.g., 64 for 64x64x64)

    Returns:
        final_grid: numpy array of shape (grid_size, grid_size, grid_size), dtype=np.float32
    """
    # Get the bounding box of the mesh
    bounds = mesh.bounds
    bounding_box_size = bounds[1] - bounds[0]
    max_dimension = max(bounding_box_size)
    scale_factor = 1.0 / max_dimension

    # Center the mesh at the origin and scale it
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(scale_factor)

    # Voxelize the normalized mesh
    pitch = 1.0 / grid_size
    voxel_grid = mesh.voxelized(pitch=pitch)

    # The voxelized method will return an object of type VoxelGrid. We have to convert it into a dense
    # matrix and then force it to have the correct shape
    matrix = voxel_grid.matrix.astype(np.float32)

    # Create an empty grid of the correct size
    final_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    min_dimensions = np.minimum(matrix.shape, [grid_size, grid_size, grid_size])

    # Copy the voxel data into the correctly sized grid
    final_grid[: min_dimensions[0], : min_dimensions[1], : min_dimensions[2]] = matrix[
        : min_dimensions[0], : min_dimensions[1], : min_dimensions[2]
    ]

    return final_grid


def voxel_to_rerun(voxel_grid, resolution=None):
    """
    Convert voxel grid to rerun format

    Args:
        voxel_grid: numpy array of shape (H, W, D)
    """

    # Reshape if needed
    if voxel_grid.ndim == 1:
        if resolution is None:
            resolution = int(np.cbrt(len(voxel_grid)))  # cube root to get resolution
        voxel_grid = voxel_grid.reshape(resolution, resolution, resolution)
    else:
        resolution = voxel_grid.shape[0]

    # Get occupied voxels
    occupied = voxel_grid > 0
    coords = np.argwhere(occupied)

    if len(coords) == 0:
        raise ValueError("No occupied voxels to display.")

    step = 1.0 / resolution
    centers = (coords + 0.5) * step - 0.5
    half_sizes = np.full((len(coords), 3), 0.5 * step, dtype=np.float32)

    return rr.Boxes3D(
        centers=centers.astype(np.float32),
        half_sizes=half_sizes,
        colors=[[80, 180, 200]],
        fill_mode="solid",
    )


def mesh_to_voxel_feature(mesh: trimesh.Trimesh, resolution: int = 16):
    """
    Convert mesh to 1D voxel feature vector

    Args:
        mesh: trimesh.Trimesh object
        resolution: int, voxel grid resolution

    Returns:
        1D numpy array of shape (resolution^3,)
    """
    # Make a copy to avoid modifying the original mesh
    mesh_copy = mesh.copy()
    voxel_grid = voxelize(mesh_copy, resolution)
    return voxel_grid.reshape(-1)


# Function to display voxel grid (16x16x16 or any resolution)
def display_voxel_grid(
    voxel_grid, resolution=None, method="matplotlib", title="Voxel Grid"
):
    """
    Display a voxel grid in 3D

    Args:
        voxel_grid: numpy array of shape (H, W, D) or flattened array of shape (H*W*D,)
        resolution: int, if voxel_grid is flattened, specify resolution (e.g., 16 for 16x16x16)
        method: str, 'matplotlib' or 'rerun' for visualization method
        title: str, title for the plot
    """
    # Reshape if needed
    if voxel_grid.ndim == 1:
        if resolution is None:
            resolution = int(np.cbrt(len(voxel_grid)))  # cube root to get resolution
        voxel_grid = voxel_grid.reshape(resolution, resolution, resolution)
    else:
        resolution = voxel_grid.shape[0]

    # Get occupied voxels
    occupied = voxel_grid > 0
    coords = np.argwhere(occupied)

    if len(coords) == 0:
        print("No occupied voxels to display.")
        return

    if method == "matplotlib":
        # Display using matplotlib 3D scatter
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Normalize coordinates to [-0.5, 0.5]
        normalized_coords = (coords + 0.5) / resolution - 0.5

        # Scatter plot
        ax.scatter(
            normalized_coords[:, 0],
            normalized_coords[:, 1],
            normalized_coords[:, 2],
            s=50,
            c="teal",
            alpha=0.6,
            edgecolors="navy",
            linewidths=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"{title} ({resolution}×{resolution}×{resolution}, {len(coords)} occupied voxels)"
        )

        # Set equal aspect ratio
        max_range = 0.6
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        plt.tight_layout()
        plt.show()

    elif method == "rerun":
        # Display using rerun (interactive)
        step = 1.0 / resolution
        centers = (coords + 0.5) * step - 0.5
        half_sizes = np.full((len(coords), 3), 0.5 * step, dtype=np.float32)

        rr.init(f"rerun_{title.replace(' ', '_')}")
        rr.log("voxels", voxel_to_rerun(voxel_grid, resolution))
        rr.notebook_show()
    else:
        raise ValueError("method must be 'matplotlib' or 'rerun'")


def mesh_to_point_cloud(mesh: trimesh.Trimesh, n_points: int = 1024):
    # Normalize mesh to unit cube (same approach as voxelize)
    mesh_copy = mesh.copy()
    bounds = mesh_copy.bounds
    bounding_box_size = bounds[1] - bounds[0]
    max_dimension = max(bounding_box_size)
    scale_factor = 1.0 / max_dimension
    
    # Center the mesh at the origin and scale it
    mesh_copy.apply_translation(-mesh_copy.centroid)
    mesh_copy.apply_scale(scale_factor)
    
    # Sample points uniformly on surface
    pts, _ = trimesh.sample.sample_surface_even(mesh_copy, n_points)
    
    # If we got fewer points than requested, pad with additional samples
    if len(pts) < n_points:
        # Sample additional points to reach n_points
        n_additional = n_points - len(pts)
        additional_pts, _ = trimesh.sample.sample_surface(mesh_copy, n_additional)
        pts = np.vstack([pts, additional_pts])
    
    # Ensure we have exactly n_points (in case we got more)
    if len(pts) > n_points:
        # Randomly select n_points if we got more
        indices = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[indices]
    
    return pts

def point_cloud_features(points: np.ndarray, radial_bins: int = 32):
    # Normalize: center and scale to unit sphere
    pts = points - points.mean(axis=0)
    max_norm = np.linalg.norm(pts, axis=1).max()
    if max_norm > 0:
        pts = pts / max_norm
    # Covariance eigenvalues (shape distribution)
    cov = np.cov(pts.T)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.maximum(evals, 1e-8))  # numerical stability
    # Radial histogram
    r = np.linalg.norm(pts, axis=1)
    hist, _ = np.histogram(r, bins=radial_bins, range=(0.0, 1.0), density=True)
    return np.concatenate([evals, hist])  # 3 + radial_bins dims


def analyze_mesh_complexity(object_path):
    """
    Analyze the complexity of a 3D mesh by reading its geometric properties
    """
    try:
        verts, faces = read_off(object_path)

        # Basic geometric properties
        num_vertices = len(verts)
        num_faces = len(faces)

        # Calculate bounding box dimensions
        min_coords = verts.min(axis=0)
        max_coords = verts.max(axis=0)
        bbox_dims = max_coords - min_coords
        bbox_volume = np.prod(bbox_dims)

        # Calculate surface area (approximate)
        # For triangular meshes, we can calculate area of each triangle
        if len(faces) > 0 and faces.shape[1] == 3:
            # Get triangle vertices
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]

            # Calculate cross products for area calculation
            cross_products = np.cross(v1 - v0, v2 - v0)
            triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
            surface_area = np.sum(triangle_areas)
        else:
            surface_area = 0

        # Calculate aspect ratio (max dimension / min dimension)
        aspect_ratio = np.max(bbox_dims) / np.min(bbox_dims) if np.min(bbox_dims) > 0 else 1

        # Calculate vertex density (vertices per unit volume)
        vertex_density = num_vertices / bbox_volume if bbox_volume > 0 else 0

        return {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'bbox_volume': bbox_volume,
            'surface_area': surface_area,
            'aspect_ratio': aspect_ratio,
            'vertex_density': vertex_density,
            'faces_per_vertex': num_faces / num_vertices if num_vertices > 0 else 0
        }
    except Exception as e:
        print(f"Error analyzing {object_path}: {e}")
        return None
