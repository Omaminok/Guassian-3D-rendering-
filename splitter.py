"""
Data Ingestion & Partitioning Module
Splits COLMAP 3D points into spatial chunks with overlapping margins
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from utils.colmap_reader import COLMAPReader


class ChunkSplitter:
    """Splits 3D points into spatial chunks for distributed processing."""
    
    def __init__(self, target_points_per_chunk: int = 50000, halo_margin_percent: float = 0.05):
        """
        Initialize chunk splitter.
        
        Args:
            target_points_per_chunk: Target points per chunk (tuned for 4GB VRAM)
            halo_margin_percent: Percentage for halo margin expansion
        """
        self.target_points_per_chunk = target_points_per_chunk
        self.halo_margin_percent = halo_margin_percent
        
    def compute_bounding_box(self, points: Dict[int, np.ndarray]) -> Tuple[float, float, float, float]:
        """
        Compute bounding box on X and Z axes only (Y = gravity axis, ignored).
        
        Args:
            points: Dict of point_id -> [x, y, z, r, g, b, error]
            
        Returns:
            Tuple of (xmin, xmax, zmin, zmax)
        """
        if not points:
            raise ValueError("No points provided")
        
        # Extract X and Z coordinates
        positions = np.array([point[:3] for point in points.values()])
        x_coords = positions[:, 0]
        z_coords = positions[:, 2]
        
        xmin, xmax = x_coords.min(), x_coords.max()
        zmin, zmax = z_coords.min(), z_coords.max()
        
        print(f"Bounding box: X[{xmin:.3f}, {xmax:.3f}], Z[{zmin:.3f}, {zmax:.3f}]")
        return xmin, xmax, zmin, zmax
    
    def calculate_grid_dimensions(self, 
                                 num_points: int, 
                                 xmin: float, xmax: float, 
                                 zmin: float, zmax: float) -> Tuple[int, int]:
        """
        Calculate optimal N x M grid for chunking.
        
        Args:
            num_points: Total number of points
            xmin, xmax: X bounding box
            zmin, zmax: Z bounding box
            
        Returns:
            Tuple of (N, M) grid dimensions
        """
        # Calculate required number of chunks
        num_chunks = max(1, int(np.ceil(num_points / self.target_points_per_chunk)))
        print(f"Target {self.target_points_per_chunk} points per chunk, need {num_chunks} chunks")
        
        # Calculate aspect ratio
        x_range = xmax - xmin
        z_range = zmax - zmin
        aspect_ratio = x_range / z_range if z_range > 0 else 1.0
        
        print(f"Aspect ratio (X/Z): {aspect_ratio:.3f}")
        
        # Find N, M such that N * M >= num_chunks and N/M approximates aspect ratio
        best_n, best_m = 1, num_chunks
        best_error = float('inf')
        
        for n in range(1, int(np.sqrt(num_chunks)) + 2):
            m = int(np.ceil(num_chunks / n))
            error = abs((n / m) - aspect_ratio)
            
            if error < best_error:
                best_error = error
                best_n, best_m = n, m
        
        print(f"Grid dimensions: {best_n} x {best_m} = {best_n * best_m} chunks")
        return best_n, best_m
    
    def create_chunk_bounding_boxes(self, 
                                  n_rows: int, n_cols: int,
                                  xmin: float, xmax: float,
                                  zmin: float, zmax: float) -> List[Dict[str, Any]]:
        """
        Create bounding boxes for each chunk with halo margins.
        
        Args:
            n_rows, n_cols: Grid dimensions
            xmin, xmax: X bounding box
            zmin, zmax: Z bounding box
            
        Returns:
            List of chunk bounding box dictionaries
        """
        chunks = []
        x_range = xmax - xmin
        z_range = zmax - zmin
        
        # Calculate cell dimensions
        cell_width = x_range / n_cols
        cell_depth = z_range / n_rows
        
        # Calculate halo margins
        halo_x = cell_width * self.halo_margin_percent
        halo_z = cell_depth * self.halo_margin_percent
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Tight bounding box
                xmin_tight = xmin + col * cell_width
                xmax_tight = xmin + (col + 1) * cell_width
                zmin_tight = zmin + row * cell_depth
                zmax_tight = zmin + (row + 1) * cell_depth
                
                # Expanded bounding box with halo
                xmin_expanded = xmin_tight - halo_x
                xmax_expanded = xmax_tight + halo_x
                zmin_expanded = zmin_tight - halo_z
                zmax_expanded = zmax_tight + halo_z
                
                # Chunk center
                center_x = (xmin_tight + xmax_tight) / 2
                center_z = (zmin_tight + zmax_tight) / 2
                
                chunk_info = {
                    'chunk_id': f'chunk_{row}_{col}',
                    'row': row,
                    'col': col,
                    'bbox_tight': [xmin_tight, xmax_tight, zmin_tight, zmax_tight],
                    'bbox_expanded': [xmin_expanded, xmax_expanded, zmin_expanded, zmax_expanded],
                    'center': [center_x, center_z],
                    'halo_margin': [halo_x, halo_z]
                }
                
                chunks.append(chunk_info)
        
        return chunks
    
    def assign_points_to_chunks(self, 
                               points: Dict[int, np.ndarray],
                               chunk_bboxes: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
        """
        Assign points to chunks based on expanded bounding boxes.
        Points can belong to multiple chunks if they're in halo margins.
        
        Args:
            points: Dict of point_id -> [x, y, z, r, g, b, error]
            chunk_bboxes: List of chunk bounding box dictionaries
            
        Returns:
            Dict mapping chunk_id -> list of point arrays
        """
        chunk_points = {chunk['chunk_id']: [] for chunk in chunk_bboxes}
        
        # Pre-compute bounding box arrays for faster checking
        bbox_arrays = {}
        for chunk in chunk_bboxes:
            chunk_id = chunk['chunk_id']
            bbox_expanded = chunk['bbox_expanded']
            bbox_arrays[chunk_id] = {
                'xmin': bbox_expanded[0],
                'xmax': bbox_expanded[1],
                'zmin': bbox_expanded[2],
                'zmax': bbox_expanded[3]
            }
        
        # Assign points to chunks
        for point_id, point_data in points.items():
            x, y, z = point_data[0], point_data[1], point_data[2]
            
            # Check which chunks this point belongs to
            for chunk_id, bbox in bbox_arrays.items():
                if (bbox['xmin'] <= x <= bbox['xmax'] and 
                    bbox['zmin'] <= z <= bbox['zmax']):
                    chunk_points[chunk_id].append(point_data)
        
        # Print statistics
        print("Point assignment statistics:")
        for chunk_id, points_list in chunk_points.items():
            print(f"  {chunk_id}: {len(points_list)} points")
        
        return chunk_points
    
    def save_chunks(self, 
                    chunk_points: Dict[str, List[np.ndarray]],
                    chunk_bboxes: List[Dict[str, Any]],
                    output_dir: str) -> None:
        """
        Save chunks to disk.
        
        Args:
            chunk_points: Dict mapping chunk_id -> list of point arrays
            chunk_bboxes: List of chunk bounding box dictionaries
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each chunk
        manifest = []
        
        for chunk in chunk_bboxes:
            chunk_id = chunk['chunk_id']
            points_list = chunk_points[chunk_id]
            
            if not points_list:
                print(f"Warning: {chunk_id} has no points!")
                continue
            
            # Convert to numpy array (XYZ coordinates only)
            points_array = np.array([point[:3] for point in points_list])
            
            # Create chunk directory
            chunk_dir = output_path / chunk_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Save points
            points_file = chunk_dir / "points.npz"
            np.savez_compressed(points_file, points=points_array)
            
            # Create metadata
            metadata = {
                'chunk_id': chunk_id,
                'row': chunk['row'],
                'col': chunk['col'],
                'bbox_tight': chunk['bbox_tight'],
                'bbox_expanded': chunk['bbox_expanded'],
                'point_count': len(points_list),
                'center': chunk['center'],
                'halo_margin': chunk['halo_margin']
            }
            
            # Save metadata
            metadata_file = chunk_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to manifest
            manifest.append({
                'chunk_id': chunk_id,
                'metadata_path': str(metadata_file),
                'points_path': str(points_file),
                'point_count': len(points_list)
            })
            
            print(f"Saved {chunk_id}: {len(points_list)} points")
        
        # Save grid manifest
        manifest_file = output_path / "grid_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Saved {len(manifest)} chunks to {output_path}")
        print(f"Grid manifest saved to {manifest_file}")
    
    def split_points(self, colmap_dir: str, output_dir: str, dry_run: bool = False) -> None:
        """
        Main function to split COLMAP points into chunks.
        
        Args:
            colmap_dir: COLMAP sparse directory
            output_dir: Output directory for chunks
            dry_run: If True, only print what would happen without writing files
        """
        print(f"Splitting points from {colmap_dir} to {output_dir}")
        print(f"Target: {self.target_points_per_chunk} points per chunk")
        print(f"Halo margin: {self.halo_margin_percent * 100}%")
        
        # Read COLMAP data
        reader = COLMAPReader(colmap_dir)
        points = reader.read_points3d()
        
        if not points:
            raise ValueError("No 3D points found in COLMAP data")
        
        # Compute bounding box
        xmin, xmax, zmin, zmax = self.compute_bounding_box(points)
        
        # Calculate grid dimensions
        n_rows, n_cols = self.calculate_grid_dimensions(len(points), xmin, xmax, zmin, zmax)
        
        # Create chunk bounding boxes
        chunk_bboxes = self.create_chunk_bounding_boxes(n_rows, n_cols, xmin, xmax, zmin, zmax)
        
        # Assign points to chunks
        chunk_points = self.assign_points_to_chunks(points, chunk_bboxes)
        
        if dry_run:
            print("\n--- DRY RUN - No files will be written ---")
            print(f"Would create {len(chunk_bboxes)} chunks:")
            for chunk in chunk_bboxes:
                chunk_id = chunk['chunk_id']
                point_count = len(chunk_points[chunk_id])
                print(f"  {chunk_id}: {point_count} points")
            return
        
        # Save chunks
        self.save_chunks(chunk_points, chunk_bboxes, output_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Split COLMAP 3D points into spatial chunks")
    parser.add_argument("--input", required=True, help="COLMAP sparse directory (e.g., ./sparse/0)")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--target_points", type=int, default=50000, 
                       help="Target points per chunk (default: 50000)")
    parser.add_argument("--halo_margin", type=float, default=0.05,
                       help="Halo margin percentage (default: 0.05)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print what would happen without writing files")
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = ChunkSplitter(
        target_points_per_chunk=args.target_points,
        halo_margin_percent=args.halo_margin
    )
    
    # Split points
    splitter.split_points(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
