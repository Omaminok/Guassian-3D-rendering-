"""
Output Assembly & Deduplication Module
Merges all chunk PLY files into one deduplicated final output
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import sys

from utils.ply_writer import PLYReader, PLYWriter, estimate_vram_usage


class GaussianStitcher:
    """Stitches together multiple chunk PLY files with deduplication."""
    
    def __init__(self, precision: int = 3):
        """
        Initialize Gaussian stitcher.
        
        Args:
            precision: Decimal precision for spatial hashing (default 3 = millimeter)
        """
        self.precision = precision
        self.ply_reader = PLYReader()
        self.ply_writer = PLYWriter()
        
    def load_all_chunks(self, results_dir: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """
        Load all chunk PLY files and metadata.
        
        Args:
            results_dir: Directory containing chunk PLY files
            
        Returns:
            Tuple of (gaussian_data_by_chunk, metadata_by_chunk)
        """
        results_path = Path(results_dir)
        ply_files = list(results_path.glob("*.ply"))
        
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {results_dir}")
        
        print(f"Loading {len(ply_files)} chunk PLY files...")
        
        gaussian_data = {}
        metadata = {}
        
        for ply_file in ply_files:
            chunk_id = ply_file.stem
            
            try:
                # Load Gaussian data
                data = self.ply_reader.read_gaussians(str(ply_file))
                gaussian_data[chunk_id] = data
                
                # Load chunk metadata if available
                chunk_dir = results_path.parent / "tasks" / chunk_id
                metadata_file = chunk_dir / "metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata[chunk_id] = json.load(f)
                else:
                    # Fallback: create minimal metadata
                    metadata[chunk_id] = {
                        'chunk_id': chunk_id,
                        'center': [0.0, 0.0]  # Default center
                    }
                
                print(f"Loaded {chunk_id}: {len(data['positions'])} Gaussians")
                
            except Exception as e:
                print(f"Error loading {ply_file}: {e}")
                continue
        
        total_gaussians_before = sum(len(data['positions']) for data in gaussian_data.values())
        print(f"Total Gaussians before deduplication: {total_gaussians_before:,}")
        
        return gaussian_data, metadata
    
    def spatial_hash_gaussians(self, gaussian_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[Tuple[float, float, float], List[Tuple[str, int]]]:
        """
        Create spatial hash of all Gaussians.
        
        Args:
            gaussian_data: Gaussian data by chunk
            
        Returns:
            Dict mapping rounded position -> list of (chunk_id, gaussian_index)
        """
        spatial_hash = defaultdict(list)
        
        print("Creating spatial hash...")
        
        for chunk_id, data in gaussian_data.items():
            positions = data['positions']
            
            for i, pos in enumerate(positions):
                # Round position to specified precision
                rounded_pos = (
                    round(pos[0], self.precision),
                    round(pos[1], self.precision),
                    round(pos[2], self.precision)
                )
                
                spatial_hash[rounded_pos].append((chunk_id, i))
        
        # Find duplicates
        duplicates = {pos: indices for pos, indices in spatial_hash.items() if len(indices) > 1}
        
        print(f"Found {len(duplicates)} duplicate positions involving "
              f"{sum(len(indices) for indices in duplicates.values())} Gaussians")
        
        return spatial_hash
    
    def resolve_duplicates(self, 
                          spatial_hash: Dict[Tuple[float, float, float], List[Tuple[str, int]]],
                          gaussian_data: Dict[str, Dict[str, np.ndarray]],
                          metadata: Dict[str, Dict[str, Any]]) -> Set[Tuple[str, int]]:
        """
        Resolve duplicate Gaussians using center distance method.
        
        Args:
            spatial_hash: Spatial hash of Gaussians
            gaussian_data: Gaussian data by chunk
            metadata: Chunk metadata
            
        Returns:
            Set of (chunk_id, gaussian_index) to keep
        """
        kept_gaussians = set()
        total_duplicates = 0
        resolved_by_distance = 0
        resolved_by_opacity = 0
        
        print("Resolving duplicates...")
        
        for pos, instances in spatial_hash.items():
            if len(instances) == 1:
                # No duplicate, keep this Gaussian
                kept_gaussians.add(instances[0])
                continue
            
            total_duplicates += len(instances)
            
            # Calculate distances to chunk centers
            distances = []
            opacities = []
            
            for chunk_id, gaussian_idx in instances:
                # Get Gaussian position
                pos_array = gaussian_data[chunk_id]['positions'][gaussian_idx]
                
                # Get chunk center
                chunk_center = np.array(metadata[chunk_id]['center'])
                
                # Calculate distance (only XZ plane as specified)
                xz_pos = pos_array[[0, 2]]  # X and Z coordinates
                xz_center = chunk_center
                
                distance = np.linalg.norm(xz_pos - xz_center)
                distances.append(distance)
                
                # Get opacity (convert from sigmoid space)
                opacity_raw = gaussian_data[chunk_id]['opacities'][gaussian_idx, 0]
                opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid
                opacities.append(opacity)
            
            # Find best Gaussian (smallest distance to chunk center)
            best_idx = np.argmin(distances)
            best_instance = instances[best_idx]
            
            # In case of tie in distance, use highest opacity
            min_distance = distances[best_idx]
            tied_indices = [i for i, d in enumerate(distances) if abs(d - min_distance) < 1e-6]
            
            if len(tied_indices) > 1:
                # Use opacity to break tie
                best_tie_idx = tied_indices[np.argmax([opacities[i] for i in tied_indices])]
                best_instance = instances[best_tie_idx]
                resolved_by_opacity += 1
            else:
                resolved_by_distance += 1
            
            kept_gaussians.add(best_instance)
        
        print(f"Resolved {total_duplicates} duplicate Gaussians:")
        print(f"  By center distance: {resolved_by_distance}")
        print(f"  By opacity (ties): {resolved_by_opacity}")
        print(f"  Kept {len(kept_gaussians)} unique Gaussians")
        
        return kept_gaussians
    
    def assemble_final_gaussians(self, 
                                 kept_gaussians: Set[Tuple[str, int]],
                                 gaussian_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Assemble final Gaussian arrays from kept instances.
        
        Args:
            kept_gaussians: Set of (chunk_id, gaussian_index) to keep
            gaussian_data: Gaussian data by chunk
            
        Returns:
            Final Gaussian data dictionary
        """
        print("Assembling final Gaussian arrays...")
        
        # Count final Gaussians
        num_final = len(kept_gaussians)
        print(f"Assembling {num_final:,} final Gaussians")
        
        # Pre-allocate arrays
        final_positions = np.zeros((num_final, 3), dtype=np.float32)
        final_sh_dc = np.zeros((num_final, 3), dtype=np.float32)
        final_opacities = np.zeros((num_final, 1), dtype=np.float32)
        final_scales = np.zeros((num_final, 3), dtype=np.float32)
        final_rotations = np.zeros((num_final, 4), dtype=np.float32)
        
        # Fill arrays
        for i, (chunk_id, gaussian_idx) in enumerate(kept_gaussians):
            data = gaussian_data[chunk_id]
            
            final_positions[i] = data['positions'][gaussian_idx]
            final_sh_dc[i] = data['sh_dc'][gaussian_idx]
            final_opacities[i] = data['opacities'][gaussian_idx]
            final_scales[i] = data['scales'][gaussian_idx]
            final_rotations[i] = data['rotations'][gaussian_idx]
        
        return {
            'positions': final_positions,
            'sh_dc': final_sh_dc,
            'opacities': final_opacities,
            'scales': final_scales,
            'rotations': final_rotations
        }
    
    def save_final_ply(self, final_gaussians: Dict[str, np.ndarray], output_path: str):
        """
        Save final merged PLY file.
        
        Args:
            final_gaussians: Final Gaussian data
            output_path: Output PLY file path
        """
        print(f"Saving final PLY to {output_path}")
        
        self.ply_writer.write_gaussians(
            positions=final_gaussians['positions'],
            sh_dc=final_gaussians['sh_dc'],
            opacities=final_gaussians['opacities'],
            scales=final_gaussians['scales'],
            rotations=final_gaussians['rotations'],
            output_path=output_path
        )
        
        # Validate output
        if self.ply_reader.validate_ply(output_path):
            print("Final PLY validation passed!")
        else:
            print("Warning: Final PLY validation failed!")
    
    def stitch_chunks(self, results_dir: str, output_path: str) -> Dict[str, Any]:
        """
        Main stitching function.
        
        Args:
            results_dir: Directory containing chunk PLY files
            output_path: Output PLY file path
            
        Returns:
            Stitching statistics
        """
        print(f"Stitching chunks from {results_dir} to {output_path}")
        
        # Load all chunk data
        gaussian_data, metadata = self.load_all_chunks(results_dir)
        
        if not gaussian_data:
            raise ValueError("No valid Gaussian data loaded")
        
        # Count total before deduplication
        total_before = sum(len(data['positions']) for data in gaussian_data.values())
        
        # Create spatial hash
        spatial_hash = self.spatial_hash_gaussians(gaussian_data)
        
        # Resolve duplicates
        kept_gaussians = self.resolve_duplicates(spatial_hash, gaussian_data, metadata)
        
        # Assemble final Gaussians
        final_gaussians = self.assemble_final_gaussians(kept_gaussians, gaussian_data)
        
        # Save final PLY
        self.save_final_ply(final_gaussians, output_path)
        
        # Calculate statistics
        total_after = len(final_gaussians['positions'])
        reduction_percent = 100 * (1 - total_after / total_before)
        vram_usage = estimate_vram_usage(total_after)
        
        # Get file size
        file_size = Path(output_path).stat().st_size / (1024**2)  # MB
        
        stats = {
            'total_gaussians_before': total_before,
            'total_gaussians_after': total_after,
            'duplicates_removed': total_before - total_after,
            'reduction_percent': reduction_percent,
            'estimated_vram_gb': vram_usage,
            'file_size_mb': file_size,
            'output_path': output_path
        }
        
        print(f"\n=== Stitching Summary ===")
        print(f"Gaussians before: {total_before:,}")
        print(f"Gaussians after: {total_after:,}")
        print(f"Duplicates removed: {total_before - total_after:,} ({reduction_percent:.1f}%)")
        print(f"Estimated VRAM to load: {vram_usage:.2f}GB")
        print(f"File size: {file_size:.1f}MB")
        print(f"Output: {output_path}")
        
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stitch chunk PLY files into final output")
    parser.add_argument("--results_dir", required=True, help="Directory containing chunk PLY files")
    parser.add_argument("--output", required=True, help="Output PLY file path")
    parser.add_argument("--precision", type=int, default=3, 
                       help="Decimal precision for spatial hashing (default: 3)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate output PLY file after stitching")
    
    args = parser.parse_args()
    
    # Create stitcher
    stitcher = GaussianStitcher(precision=args.precision)
    
    # Stitch chunks
    try:
        stats = stitcher.stitch_chunks(args.results_dir, args.output)
        
        # Validate if requested
        if args.validate:
            print("\nValidating output PLY...")
            reader = PLYReader()
            if reader.validate_ply(args.output):
                print("Output validation successful!")
            else:
                print("Output validation failed!")
                sys.exit(1)
        
        print("Stitching completed successfully!")
        
    except Exception as e:
        print(f"Stitching failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
