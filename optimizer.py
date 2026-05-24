"""
Camera Frustum Culling Module
Filters cameras that actually see each chunk to reduce training workload
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from utils.colmap_reader import COLMAPReader


class CameraOptimizer:
    """
    Camera frustum culling for 3D Gaussian Splatting chunks.
    Note: This is "optimizer" in the sense of optimizing the camera set, not gradient optimization.
    """
    
    def __init__(self, margin_percent: float = 0.1):
        """
        Initialize camera optimizer.
        
        Args:
            margin_percent: Margin for frustum culling (default 10%)
        """
        self.margin_percent = margin_percent
        
    def load_chunk_metadata(self, chunk_dir: str) -> Dict[str, Any]:
        """
        Load chunk metadata.
        
        Args:
            chunk_dir: Path to chunk directory
            
        Returns:
            Chunk metadata dictionary
        """
        chunk_path = Path(chunk_dir)
        metadata_file = chunk_path / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_chunk_points(self, chunk_dir: str) -> np.ndarray:
        """
        Load chunk points for Y-axis bounding box calculation.
        
        Args:
            chunk_dir: Path to chunk directory
            
        Returns:
            Points array of shape (N, 3)
        """
        chunk_path = Path(chunk_dir)
        points_file = chunk_path / "points.npz"
        
        if not points_file.exists():
            raise FileNotFoundError(f"Chunk points not found: {points_file}")
        
        data = np.load(points_file)
        points = data['points']
        
        return points
    
    def build_chunk_bounding_box(self, metadata: Dict[str, Any], points: np.ndarray) -> np.ndarray:
        """
        Build 3D bounding box for chunk.
        
        Args:
            metadata: Chunk metadata
            points: Chunk points array
            
        Returns:
            8 corners of 3D bounding box as (8, 3) array
        """
        # Use bbox_expanded for X and Z from metadata
        bbox_expanded = metadata['bbox_expanded']
        xmin, xmax, zmin, zmax = bbox_expanded
        
        # Calculate Y bounds from actual points with 10% expansion
        y_coords = points[:, 1]
        ymin, ymax = y_coords.min(), y_coords.max()
        y_range = ymax - ymin
        ymin_expanded = ymin - y_range * 0.1
        ymax_expanded = ymax + y_range * 0.1
        
        # Generate 8 corners of the bounding box
        corners = np.array([
            [xmin, ymin_expanded, zmin],  # 0
            [xmax, ymin_expanded, zmin],  # 1
            [xmin, ymax_expanded, zmin],  # 2
            [xmax, ymax_expanded, zmin],  # 3
            [xmin, ymin_expanded, zmax],  # 4
            [xmax, ymin_expanded, zmax],  # 5
            [xmin, ymax_expanded, zmax],  # 6
            [xmax, ymax_expanded, zmax],  # 7
        ])
        
        return corners
    
    def transform_points_to_camera_space(self, 
                                       corners: np.ndarray, 
                                       R: np.ndarray, 
                                       t: np.ndarray) -> np.ndarray:
        """
        Transform world points to camera space.
        
        Args:
            corners: World points (N, 3)
            R: Rotation matrix (3, 3)
            t: Translation vector (3,)
            
        Returns:
            Camera space points (N, 3)
        """
        # Transform: p_cam = R @ p_world + t
        camera_points = (R @ corners.T).T + t
        return camera_points
    
    def project_to_image_space(self, 
                             camera_points: np.ndarray,
                             fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """
        Project camera space points to image coordinates.
        
        Args:
            camera_points: Camera space points (N, 3)
            fx, fy: Focal lengths
            cx, cy: Principal points
            
        Returns:
            Image coordinates (N, 2)
        """
        # Handle points behind camera
        z = camera_points[:, 2]
        valid_mask = z > 0
        
        image_points = np.full((len(camera_points), 2), -1.0)
        
        if np.any(valid_mask):
            x = camera_points[valid_mask, 0]
            y = camera_points[valid_mask, 1]
            z_valid = z[valid_mask]
            
            u = fx * x / z_valid + cx
            v = fy * y / z_valid + cy
            
            image_points[valid_mask, 0] = u
            image_points[valid_mask, 1] = v
        
        return image_points
    
    def is_camera_visible(self, 
                         corners: np.ndarray,
                         R: np.ndarray, t: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         width: int, height: int) -> bool:
        """
        Check if camera can see the chunk bounding box.
        
        Args:
            corners: 3D bounding box corners (8, 3)
            R, t: Camera extrinsics
            fx, fy, cx, cy: Camera intrinsics
            width, height: Image dimensions
            
        Returns:
            True if camera can see the chunk
        """
        # Transform to camera space
        camera_points = self.transform_points_to_camera_space(corners, R, t)
        
        # Check if all points are behind camera
        if np.all(camera_points[:, 2] <= 0):
            return False
        
        # Project to image space
        image_points = self.project_to_image_space(camera_points, fx, fy, cx, cy)
        
        # Check bounds with margin
        margin = self.margin_percent
        x_min, x_max = -width * margin, width * (1 + margin)
        y_min, y_max = -height * margin, height * (1 + margin)
        
        # Check if all projected points are outside image bounds
        outside_mask = ((image_points[:, 0] < x_min) | (image_points[:, 0] > x_max) |
                       (image_points[:, 1] < y_min) | (image_points[:, 1] > y_max))
        
        # If all points are outside bounds, camera doesn't see the chunk
        if np.all(outside_mask):
            return False
        
        return True
    
    def process_chunk(self, 
                     chunk_dir: str,
                     cameras: Dict[int, Dict],
                     images: Dict[int, Dict],
                     images_dir: str) -> List[Dict[str, Any]]:
        """
        Process a single chunk to find visible cameras.
        
        Args:
            chunk_dir: Path to chunk directory
            cameras: COLMAP cameras data
            images: COLMAP images data
            images_dir: Directory containing the actual image files
            
        Returns:
            List of visible camera dictionaries
        """
        print(f"Processing chunk: {chunk_dir}")
        
        # Load chunk data
        metadata = self.load_chunk_metadata(chunk_dir)
        points = self.load_chunk_points(chunk_dir)
        
        # Build 3D bounding box
        corners = self.build_chunk_bounding_box(metadata, points)
        
        # Find visible cameras
        visible_cameras = []
        total_cameras = len(images)
        
        for image_id, image_data in images.items():
            camera_id = image_data['camera_id']
            
            if camera_id not in cameras:
                continue
            
            camera_data = cameras[camera_id]
            
            # Get camera intrinsics
            try:
                fx, fy, cx, cy = COLMAPReader().get_pinhole_intrinsics(camera_data['params'])
            except ValueError:
                # Skip non-pinhole cameras
                continue
            
            # Get camera extrinsics
            qvec = image_data['qvec']
            tvec = image_data['tvec']
            
            # Convert quaternion to rotation matrix
            R = COLMAPReader().quaternion_to_rotation_matrix(qvec)
            
            # Transform translation to camera coordinate system
            t = R @ tvec
            
            # Check visibility
            if self.is_camera_visible(corners, R, t, fx, fy, cx, cy, 
                                    camera_data['width'], camera_data['height']):
                # Build camera dictionary
                camera_dict = {
                    'image_id': image_id,
                    'image_name': image_data['name'],
                    'width': camera_data['width'],
                    'height': camera_data['height'],
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'R': R.tolist(),
                    't': t.tolist(),
                    'image_path': str(Path(images_dir) / image_data['name'])
                }
                
                visible_cameras.append(camera_dict)
        
        print(f"Found {len(visible_cameras)} visible cameras out of {total_cameras} "
              f"({100 * len(visible_cameras) / total_cameras:.1f}%)")
        
        return visible_cameras
    
    def save_cameras(self, cameras: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save visible cameras to JSON file.
        
        Args:
            cameras: List of camera dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(cameras, f, indent=2)
        
        print(f"Saved {len(cameras)} cameras to {output_file}")
    
    def optimize_all_chunks(self, 
                           colmap_dir: str,
                           tasks_dir: str,
                           images_dir: str,
                           dry_run: bool = False) -> None:
        """
        Optimize cameras for all chunks.
        
        Args:
            colmap_dir: COLMAP sparse directory
            tasks_dir: Directory containing chunk data
            images_dir: Directory containing image files
            dry_run: If True, only print what would happen
        """
        print(f"Optimizing cameras for chunks in {tasks_dir}")
        print(f"Using COLMAP data from {colmap_dir}")
        print(f"Using images from {images_dir}")
        
        # Read COLMAP data
        reader = COLMAPReader(colmap_dir)
        cameras = reader.read_cameras()
        images = reader.read_images()
        
        print(f"Loaded {len(cameras)} cameras and {len(images)} images")
        
        # Find all chunk directories
        tasks_path = Path(tasks_dir)
        chunk_dirs = [d for d in tasks_path.iterdir() 
                     if d.is_dir() and d.name.startswith('chunk_')]
        
        chunk_dirs.sort()  # Process in order
        print(f"Found {len(chunk_dirs)} chunks")
        
        # Process each chunk
        total_cameras_before = 0
        total_cameras_after = 0
        
        for chunk_dir in chunk_dirs:
            if dry_run:
                print(f"Would process: {chunk_dir}")
                total_cameras_before += len(images)
                continue
            
            try:
                visible_cameras = self.process_chunk(
                    str(chunk_dir), cameras, images, images_dir
                )
                
                # Save cameras
                cameras_file = chunk_dir / "cameras.json"
                self.save_cameras(visible_cameras, str(cameras_file))
                
                total_cameras_before += len(images)
                total_cameras_after += len(visible_cameras)
                
            except Exception as e:
                print(f"Error processing {chunk_dir}: {e}")
                continue
        
        if not dry_run:
            reduction_percent = 100 * (1 - total_cameras_after / total_cameras_before)
            print(f"\nCamera culling summary:")
            print(f"  Total cameras before: {total_cameras_before}")
            print(f"  Total cameras after: {total_cameras_after}")
            print(f"  Reduction: {reduction_percent:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera frustum culling for 3DGS chunks")
    parser.add_argument("--colmap_dir", required=True, 
                       help="COLMAP sparse directory (e.g., ./sparse/0)")
    parser.add_argument("--tasks_dir", required=True,
                       help="Directory containing chunk data")
    parser.add_argument("--images_dir", required=True,
                       help="Directory containing image files")
    parser.add_argument("--margin", type=float, default=0.1,
                       help="Frustum culling margin percentage (default: 0.1)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print what would happen without writing files")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = CameraOptimizer(margin_percent=args.margin)
    
    # Optimize all chunks
    optimizer.optimize_all_chunks(
        args.colmap_dir, args.tasks_dir, args.images_dir, args.dry_run
    )


if __name__ == "__main__":
    main()
