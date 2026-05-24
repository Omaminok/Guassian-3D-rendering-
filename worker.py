"""
Rendering & Optimization Module
Trains 3D Gaussians on individual chunks using gsplat

For distributed mode, use distributed/master_orchestrator.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import math

try:
    import gsplat
except ImportError:
    print("Warning: gsplat not installed. Please install with: pip install gsplat")
    gsplat = None

from utils.vram_guard import VRAMGuard, ChunkOOMError
from utils.ply_writer import PLYWriter

# Optional Redis import for distributed progress reporting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class GaussianWorker:
    """Worker for training 3D Gaussians on a single chunk."""
    
    def __init__(self, device: torch.device, redis_conn: Optional[Any] = None):
        """
        Initialize Gaussian worker.
        
        Args:
            device: PyTorch device for training
            redis_conn: Optional Redis connection for progress reporting
        """
        self.device = device
        self.vram_guard = VRAMGuard(device)
        self.ply_writer = PLYWriter()
        self.redis_conn = redis_conn
        
    def load_chunk_data(self, chunk_dir: str) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load chunk points, metadata, and cameras.
        
        Args:
            chunk_dir: Path to chunk directory
            
        Returns:
            Tuple of (points, metadata, cameras)
        """
        chunk_path = Path(chunk_dir)
        
        # Load points
        points_file = chunk_path / "points.npz"
        if not points_file.exists():
            raise FileNotFoundError(f"Points file not found: {points_file}")
        
        points_data = np.load(points_file)
        points = points_data['points']  # (N, 3)
        
        # Load metadata
        metadata_file = chunk_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load cameras
        cameras_file = chunk_path / "cameras.json"
        if not cameras_file.exists():
            raise FileNotFoundError(f"Cameras file not found: {cameras_file}")
        
        with open(cameras_file, 'r') as f:
            cameras = json.load(f)
        
        print(f"Loaded {len(points)} points and {len(cameras)} cameras for {metadata['chunk_id']}")
        return points, metadata, cameras
    
    def initialize_gaussians(self, points: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Initialize Gaussian parameters from points.
        
        Args:
            points: Initial point positions (N, 3)
            
        Returns:
            Dictionary of Gaussian parameters
        """
        num_points = len(points)
        
        # Positions (float32)
        positions = torch.from_numpy(points).float().to(self.device)
        
        # Opacities (float32, sigmoid space, initialized to 0.1)
        opacities = torch.full((num_points, 1), 0.1, dtype=torch.float32, device=self.device)
        opacities = torch.logit(opacities)  # Convert to sigmoid space
        
        # Scales (float32, log space)
        # Initialize based on nearest neighbor distance
        if num_points > 1:
            # Compute nearest neighbor distances
            with torch.no_grad():
                distances = torch.cdist(positions, positions)
                distances.fill_diagonal(float('inf'))
                nn_distances = distances.min(dim=1)[0]
                scale_init = (nn_distances * 0.5).unsqueeze(1)
            scales = torch.log(scale_init.clamp(min=0.001))
        else:
            scales = torch.full((num_points, 3), -3.0, dtype=torch.float32, device=self.device)
        
        # Rotations (float32, quaternions, identity)
        rotations = torch.zeros((num_points, 4), dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.0  # Identity quaternion [1, 0, 0, 0]
        
        # SH coefficients (float16, degree 1 only)
        # Initialize with small random values
        sh_dc = torch.randn(num_points, 3, dtype=torch.float16, device=self.device) * 0.1
        
        return {
            'means': positions,           # (N, 3) float32
            'opacities': opacities,      # (N, 1) float32
            'scales': scales,            # (N, 3) float32
            'rotations': rotations,      # (N, 4) float32
            'sh_dc': sh_dc               # (N, 3) float16
        }
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (800, 800)) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            target_size: Target size (max dimension)
            
        Returns:
            Tuple of (image_tensor, camera_info)
        """
        # Load image
        image = Image.open(image_path)
        
        # Resize to target size (maintain aspect ratio)
        w, h = image.size
        max_dim = max(w, h)
        if max_dim > target_size[0]:
            scale = target_size[0] / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, {'original_size': (w, h), 'processed_size': image_tensor.shape[:2]}
    
    def setup_camera(self, camera_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Setup camera parameters for rendering.
        
        Args:
            camera_info: Camera dictionary from optimizer
            
        Returns:
            Camera parameters dictionary
        """
        # Extract camera parameters
        width = camera_info['width']
        height = camera_info['height']
        fx, fy = camera_info['fx'], camera_info['fy']
        cx, cy = camera_info['cx'], camera_info['cy']
        
        # Build camera extrinsics
        R = torch.tensor(camera_info['R'], dtype=torch.float32, device=self.device)
        t = torch.tensor(camera_info['t'], dtype=torch.float32, device=self.device)
        
        # Build view matrix (world to camera)
        view_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t
        view_matrix[3, 3] = 1.0
        
        # Build projection matrix
        projection_matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        projection_matrix[0, 0] = 2 * fx / width
        projection_matrix[1, 1] = 2 * fy / height
        projection_matrix[0, 2] = (2 * cx - width) / width
        projection_matrix[1, 2] = (2 * cy - height) / height
        projection_matrix[2, 2] = (1.0 + 0.01) / (1.0 - 0.01)  # near=0.01, far=100
        projection_matrix[2, 3] = -2 * 0.01 * 1.0 / (1.0 - 0.01)
        projection_matrix[3, 2] = 1.0
        
        return {
            'view_matrix': view_matrix,
            'projection_matrix': projection_matrix,
            'width': width,
            'height': height
        }
    
    def render_gaussians(self, gaussians: Dict[str, torch.Tensor], camera: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Render Gaussians to image.
        
        Args:
            gaussians: Gaussian parameters
            camera: Camera parameters
            
        Returns:
            Rendered image tensor
        """
        if gsplat is None:
            # Fallback: return black image
            height, width = camera['height'], camera['width']
            return torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        
        # Convert Gaussian parameters to gsplat format
        means = gaussians['means']
        scales = torch.exp(gaussians['scales'])  # Convert from log space
        rotations = gaussians['rotations']
        opacities = torch.sigmoid(gaussians['opacities'])  # Convert from sigmoid space
        
        # Convert SH from float16 to float32 for rendering
        sh_dc = gaussians['sh_dc'].float()
        
        # Build camera parameters
        height, width = camera['height'], camera['width']
        fx = camera['projection_matrix'][0, 0] * width / 2
        fy = camera['projection_matrix'][1, 1] * height / 2
        cx = camera['projection_matrix'][0, 2] * width / 2 + width / 2
        cy = camera['projection_matrix'][1, 2] * height / 2 + height / 2
        
        # Render using gsplat
        try:
            image, _, _ = gsplat.rasterization(
                means=means,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                shs=sh_dc.unsqueeze(1),  # Add degree dimension
                viewmats=torch.inverse(camera['view_matrix']).unsqueeze(0),
                Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device).unsqueeze(0),
                width=width,
                height=height,
            )
            
            # image is (1, 3, H, W), convert to (H, W, 3)
            image = image[0].permute(1, 2, 0)
            
        except Exception as e:
            print(f"gsplat rendering failed: {e}")
            # Fallback to black image
            image = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        
        return image
    
    def compute_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            rendered: Rendered image
            target: Target image
            
        Returns:
            Loss tensor
        """
        # Resize target to match rendered if needed
        if rendered.shape[:2] != target.shape[:2]:
            target = F.interpolate(target.permute(2, 0, 1).unsqueeze(0), 
                                size=rendered.shape[:2], mode='bilinear').squeeze(0).permute(1, 2, 0)
        
        # L1 loss
        l1_loss = F.l1_loss(rendered, target)
        
        # SSIM loss (simplified)
        mu_x = rendered.mean(dim=[0, 1])
        mu_y = target.mean(dim=[0, 1])
        sigma_x = rendered.std(dim=[0, 1])
        sigma_y = target.std(dim=[0, 1])
        sigma_xy = ((rendered - mu_x) * (target - mu_y)).mean(dim=[0, 1])
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
        ssim_loss = 1 - ssim.mean()
        
        # Combined loss
        total_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        return total_loss
    
    def adaptive_density_control(self, 
                                gaussians: Dict[str, torch.Tensor],
                                iteration: int,
                                grad_norms: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform adaptive density control (clone, split, prune).
        
        Args:
            gaussians: Gaussian parameters
            iteration: Current iteration
            grad_norms: Gradient norms for densification
            
        Returns:
            Updated Gaussian parameters
        """
        # Only densify between iterations 500 and 4000, every 100 iterations
        if iteration < 500 or iteration > 4000 or iteration % 100 != 0:
            return gaussians
        
        num_gaussians = len(gaussians['means'])
        
        # Skip if too many Gaussians (VRAM safety)
        if num_gaussians > 200000:
            print(f"Skipping densification: too many Gaussians ({num_gaussians})")
            return gaussians
        
        # Get opacity values
        opacities = torch.sigmoid(gaussians['opacities'])
        scales = torch.exp(gaussians['scales'])
        
        # Prune low-opacity Gaussians
        prune_mask = opacities.squeeze() >= 0.005
        
        # Clone condition: high gradient norm, small scale
        if grad_norms is not None:
            clone_mask = (grad_norms > 0.0002) & (scales.max(dim=1)[0] < 0.01)
            
            # Split condition: high gradient norm, large scale
            split_mask = (grad_norms > 0.0002) & (scales.max(dim=1)[0] > 0.01)
            
            # Apply operations
            new_gaussians = {}
            for key, tensor in gaussians.items():
                if key == 'sh_dc':
                    # Keep float16 for SH
                    new_gaussians[key] = tensor[prune_mask]
                else:
                    new_gaussians[key] = tensor[prune_mask]
            
            # Clone Gaussians
            if torch.any(clone_mask):
                clone_indices = torch.where(clone_mask & prune_mask)[0]
                for idx in clone_indices:
                    for key, tensor in new_gaussians.items():
                        if key == 'means':
                            # Offset slightly
                            offset = torch.randn(3, device=self.device) * 0.01
                            new_tensor = tensor[idx] + offset
                            new_gaussians[key] = torch.cat([new_gaussians[key], new_tensor.unsqueeze(0)])
                        else:
                            new_gaussians[key] = torch.cat([new_gaussians[key], tensor[idx].unsqueeze(0)])
            
            # Split Gaussians
            if torch.any(split_mask):
                split_indices = torch.where(split_mask & prune_mask)[0]
                for idx in split_indices:
                    for key, tensor in new_gaussians.items():
                        if key == 'means':
                            # Sample from Gaussian distribution
                            mean = tensor[idx]
                            scale = torch.exp(gaussians['scales'][idx])
                            offset1 = torch.randn(3, device=self.device) * scale * 0.5
                            offset2 = torch.randn(3, device=self.device) * scale * 0.5
                            new_tensor1 = mean + offset1
                            new_tensor2 = mean + offset2
                            new_gaussians[key] = torch.cat([new_gaussians[key], 
                                                          new_tensor1.unsqueeze(0), 
                                                          new_tensor2.unsqueeze(0)])
                        elif key == 'scales':
                            # Reduce scale for split Gaussians
                            new_scale = tensor[idx] - np.log(2)
                            new_gaussians[key] = torch.cat([new_gaussians[key], 
                                                          new_scale.unsqueeze(0), 
                                                          new_scale.unsqueeze(0)])
                        else:
                            new_gaussians[key] = torch.cat([new_gaussians[key], 
                                                          tensor[idx].unsqueeze(0), 
                                                          tensor[idx].unsqueeze(0)])
            
            gaussians = new_gaussians
            
        print(f"After densification: {len(gaussians['means'])} Gaussians")
        
        return gaussians
    
    def publish_progress(self, chunk_id: str, iteration: int, total_iterations: int):
        """
        Publish training progress to Redis for distributed monitoring.
        
        Args:
            chunk_id: Chunk identifier
            iteration: Current iteration number
            total_iterations: Total number of iterations
        """
        if self.redis_conn and REDIS_AVAILABLE:
            try:
                progress_data = {
                    "chunk_id": chunk_id,
                    "iteration": iteration,
                    "total_iterations": total_iterations,
                    "percentage": (iteration / total_iterations) * 100,
                    "timestamp": time.time()
                }
                
                # Publish progress with TTL of 2 minutes
                self.redis_conn.setex(
                    f"progress:{chunk_id}", 
                    120,  # 2 minutes TTL
                    json.dumps(progress_data)
                )
                
                # Also publish to a channel for real-time updates
                self.redis_conn.publish(
                    "progress_updates",
                    json.dumps(progress_data)
                )
                
            except Exception as e:
                # Don't let progress publishing errors interrupt training
                print(f"Warning: Failed to publish progress: {e}")
    
    def train_chunk(self, 
                   chunk_dir: str,
                   config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train 3D Gaussians on a single chunk.
        
        Args:
            chunk_dir: Path to chunk directory
            config: Training configuration
            
        Returns:
            Training results dictionary
        """
        chunk_id = Path(chunk_dir).name
        print(f"Starting training for {chunk_id}")
        
        start_time = time.time()
        
        try:
            # Load data
            points, metadata, cameras = self.load_chunk_data(chunk_dir)
            
            # Initialize Gaussians
            gaussians = self.initialize_gaussians(points)
            
            # Setup optimizers
            optimizer = torch.optim.Adam([
                {'params': gaussians['means'], 'lr': config['lr_positions']},
                {'params': gaussians['opacities'], 'lr': config['lr_opacities']},
                {'params': gaussians['scales'], 'lr': config['lr_scales']},
                {'params': gaussians['rotations'], 'lr': config['lr_rotations']},
                {'params': gaussians['sh_dc'], 'lr': config['lr_colors']}
            ])
            
            # Training loop
            num_iterations = config['num_iterations']
            densify_interval = config['densify_interval']
            
            for iteration in range(num_iterations):
                # Select random camera
                camera_idx = np.random.randint(len(cameras))
                camera_info = cameras[camera_idx]
                
                # Setup camera
                camera = self.setup_camera(camera_info)
                
                # Load image
                image_tensor, _ = self.load_image(camera_info['image_path'])
                
                # Render
                rendered = self.render_gaussians(gaussians, camera)
                
                # Compute loss
                loss = self.compute_loss(rendered, image_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norms for densification
                grad_norms = None
                if iteration % densify_interval == 0 and iteration >= 500:
                    grad_norms = torch.norm(gaussians['means'].grad, dim=1)
                
                optimizer.step()
                
                # Adaptive density control
                gaussians = self.adaptive_density_control(gaussians, iteration, grad_norms)
                
                # Update optimizer with new parameters
                optimizer = torch.optim.Adam([
                    {'params': gaussians['means'], 'lr': config['lr_positions']},
                    {'params': gaussians['opacities'], 'lr': config['lr_opacities']},
                    {'params': gaussians['scales'], 'lr': config['lr_scales']},
                    {'params': gaussians['rotations'], 'lr': config['lr_rotations']},
                    {'params': gaussians['sh_dc'], 'lr': config['lr_colors']}
                ])
                
                # Memory cleanup
                if iteration % 100 == 0:
                    self.vram_guard.aggressive_cleanup()
                
                # Publish progress every 100 iterations
                if iteration % 100 == 0:
                    self.publish_progress(chunk_id, iteration, num_iterations)
                
                # Progress logging
                if iteration % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.6f}, "
                          f"Gaussians: {len(gaussians['means'])}, Time: {elapsed:.1f}s")
            
            # Save results
            output_path = Path(config['output_dir']) / f"{chunk_id}.ply"
            
            # Convert SH back to float32 for saving
            sh_dc_float32 = gaussians['sh_dc'].float()
            
            self.ply_writer.write_gaussians(
                positions=gaussians['means'].cpu().numpy(),
                sh_dc=sh_dc_float32.cpu().numpy(),
                opacities=gaussians['opacities'].cpu().numpy(),
                scales=gaussians['scales'].cpu().numpy(),
                rotations=gaussians['rotations'].cpu().numpy(),
                output_path=str(output_path)
            )
            
            training_time = time.time() - start_time
            
            results = {
                'chunk_id': chunk_id,
                'status': 'completed',
                'num_gaussians': len(gaussians['means']),
                'num_iterations': num_iterations,
                'training_time': training_time,
                'output_path': str(output_path)
            }
            
            print(f"Training completed for {chunk_id}: {len(gaussians['means'])} Gaussians, "
                  f"{training_time:.1f}s")
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            self.vram_guard.aggressive_cleanup()
            raise ChunkOOMError(f"OOM in {chunk_id}: {e}")
            
        except Exception as e:
            return {
                'chunk_id': chunk_id,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }


def train_chunk_distributed(chunk_id: str, chunk_dir: str, output_dir: str, 
                           config: Dict[str, Any], redis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Distributed-compatible wrapper for training chunks.
    This function is designed to be called by RQ tasks.
    
    Args:
        chunk_id: Chunk identifier
        chunk_dir: Path to chunk directory
        output_dir: Output directory for results
        config: Training configuration
        redis_config: Optional Redis configuration for progress reporting
        
    Returns:
        Training results dictionary
    """
    # Setup Redis connection if provided
    redis_conn = None
    if redis_config and REDIS_AVAILABLE:
        try:
            redis_conn = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                decode_responses=True
            )
        except Exception as e:
            print(f"Warning: Failed to connect to Redis for progress: {e}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create worker with Redis connection
    worker = GaussianWorker(device, redis_conn)
    
    # Prepare configuration
    worker_config = config.copy()
    worker_config['output_dir'] = output_dir
    
    # Train chunk
    return worker.train_chunk(chunk_dir, worker_config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train 3D Gaussians on a chunk")
    parser.add_argument("--chunk_dir", required=True, help="Path to chunk directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--num_iterations", type=int, default=3000, help="Number of training iterations")
    parser.add_argument("--lr_positions", type=float, default=1.6e-4, help="Learning rate for positions")
    parser.add_argument("--lr_opacities", type=float, default=1e-2, help="Learning rate for opacities")
    parser.add_argument("--lr_scales", type=float, default=1e-3, help="Learning rate for scales")
    parser.add_argument("--lr_rotations", type=float, default=1e-3, help="Learning rate for rotations")
    parser.add_argument("--lr_colors", type=float, default=5e-3, help="Learning rate for colors")
    parser.add_argument("--densify_interval", type=int, default=100, help="Densification interval")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create worker
    worker = GaussianWorker(device)
    
    # Training configuration
    config = {
        'num_iterations': args.num_iterations,
        'lr_positions': args.lr_positions,
        'lr_opacities': args.lr_opacities,
        'lr_scales': args.lr_scales,
        'lr_rotations': args.lr_rotations,
        'lr_colors': args.lr_colors,
        'densify_interval': args.densify_interval,
        'output_dir': args.output_dir
    }
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train chunk
    results = worker.train_chunk(args.chunk_dir, config)
    
    # Print results
    print(f"Training results: {results}")


if __name__ == "__main__":
    main()
