"""
Distributed Systems Management Module
Manages and executes all chunk training jobs with fault tolerance
"""

import sqlite3
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import heapq

from worker import GaussianWorker
from utils.vram_guard import VRAMGuard, ChunkOOMError


@dataclass
class ChunkConfig:
    """Configuration for chunk training."""
    chunk_id: str
    max_gaussians: int = 200000
    densify_interval: int = 100
    num_iters: int = 3000
    batch_size: int = 1
    attempt: int = 0
    max_attempts: int = 3


class DatabaseManager:
    """Manages SQLite state database for chunk tracking."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'PENDING',
                    attempt INTEGER DEFAULT 0,
                    batch_size INTEGER DEFAULT 1,
                    max_gaussians INTEGER DEFAULT 200000,
                    densify_interval INTEGER DEFAULT 100,
                    num_iters INTEGER DEFAULT 3000,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_chunk(self, chunk_id: str, config: ChunkConfig):
        """Add or update chunk in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, status, attempt, batch_size, max_gaussians, 
                 densify_interval, num_iters)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (chunk_id, 'PENDING', config.attempt, config.batch_size,
                  config.max_gaussians, config.densify_interval, config.num_iters))
            conn.commit()
    
    def get_chunk_status(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk status from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def get_pending_chunks(self) -> List[str]:
        """Get list of pending chunk IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT chunk_id FROM chunks WHERE status = "PENDING"')
            return [row[0] for row in cursor.fetchall()]
    
    def update_chunk_status(self, chunk_id: str, status: str, error_message: str = None):
        """Update chunk status."""
        with sqlite3.connect(self.db_path) as conn:
            if status == 'PROCESSING':
                conn.execute('''
                    UPDATE chunks SET status = ?, started_at = CURRENT_TIMESTAMP 
                    WHERE chunk_id = ?
                ''', (status, chunk_id))
            elif status == 'COMPLETED':
                conn.execute('''
                    UPDATE chunks SET status = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE chunk_id = ?
                ''', (status, chunk_id))
            elif status == 'FAILED':
                conn.execute('''
                    UPDATE chunks SET status = ?, error_message = ? 
                    WHERE chunk_id = ?
                ''', (status, error_message, chunk_id))
            else:
                conn.execute('UPDATE chunks SET status = ? WHERE chunk_id = ?', 
                           (status, chunk_id))
            conn.commit()
    
    def increment_chunk_attempt(self, chunk_id: str, new_config: ChunkConfig):
        """Increment chunk attempt and update config."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE chunks SET 
                    status = 'PENDING',
                    attempt = ?,
                    max_gaussians = ?,
                    densify_interval = ?,
                    num_iters = ?
                WHERE chunk_id = ?
            ''', (new_config.attempt, new_config.max_gaussians, 
                  new_config.densify_interval, new_config.num_iters, chunk_id))
            conn.commit()
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks ORDER BY chunk_id')
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def reset_processing_chunks(self):
        """Reset PROCESSING chunks to PENDING (crash recovery)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE chunks SET status = "PENDING" WHERE status = "PROCESSING"')
            conn.commit()


class ChunkPrioritizer:
    """Manages chunk processing priority based on size."""
    
    def __init__(self, manifest: List[Dict[str, Any]]):
        """
        Initialize prioritizer.
        
        Args:
            manifest: Grid manifest with chunk metadata
        """
        self.manifest = manifest
        self.chunks_by_size = {}
        
        # Build size lookup
        for chunk_info in manifest:
            chunk_id = chunk_info['chunk_id']
            point_count = chunk_info['point_count']
            self.chunks_by_size[chunk_id] = point_count
    
    def get_priority_queue(self, pending_chunks: List[str]) -> List[Tuple[int, str]]:
        """
        Get priority queue (smallest chunks first).
        
        Args:
            pending_chunks: List of pending chunk IDs
            
        Returns:
            List of (priority, chunk_id) tuples
        """
        priority_queue = []
        
        for chunk_id in pending_chunks:
            if chunk_id in self.chunks_by_size:
                point_count = self.chunks_by_size[chunk_id]
                # Use point count as priority (smaller = higher priority)
                priority_queue.append((point_count, chunk_id))
        
        heapq.heapify(priority_queue)
        return priority_queue


class PipelineOrchestrator:
    """Main orchestrator for 3D Gaussian Splatting pipeline."""
    
    def __init__(self, 
                 tasks_dir: str,
                 images_dir: str,
                 output_dir: str,
                 device: str = 'cuda'):
        """
        Initialize orchestrator.
        
        Args:
            tasks_dir: Directory containing chunk tasks
            images_dir: Directory containing images
            output_dir: Directory for output results
            device: PyTorch device
        """
        self.tasks_dir = Path(tasks_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Initialize components
        self.db_manager = DatabaseManager(str(self.output_dir / "state.db"))
        self.worker = GaussianWorker(device)
        
        # Setup logging
        self.setup_logging()
        
        # Load manifest
        self.manifest = self.load_manifest()
        self.prioritizer = ChunkPrioritizer(self.manifest)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_manifest(self) -> List[Dict[str, Any]]:
        """Load grid manifest."""
        manifest_file = self.tasks_dir / "grid_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Grid manifest not found: {manifest_file}")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        self.logger.info(f"Loaded manifest with {len(manifest)} chunks")
        return manifest
    
    def initialize_chunks(self):
        """Initialize chunks in database."""
        self.logger.info("Initializing chunks in database...")
        
        # Reset any processing chunks (crash recovery)
        self.db_manager.reset_processing_chunks()
        
        # Add all chunks from manifest
        for chunk_info in self.manifest:
            chunk_id = chunk_info['chunk_id']
            
            # Check if chunk already exists
            existing = self.db_manager.get_chunk_status(chunk_id)
            if existing is None:
                # Add new chunk
                config = ChunkConfig(chunk_id=chunk_id)
                self.db_manager.add_chunk(chunk_id, config)
            elif existing['status'] == 'COMPLETED':
                self.logger.info(f"Skipping completed chunk: {chunk_id}")
        
        self.logger.info("Chunk initialization completed")
    
    def create_chunk_config(self, chunk_id: str, attempt: int) -> ChunkConfig:
        """
        Create configuration for chunk training attempt.
        
        Args:
            chunk_id: Chunk identifier
            attempt: Attempt number
            
        Returns:
            Chunk configuration
        """
        # Base configuration
        max_gaussians = 200000
        densify_interval = 100
        num_iters = 3000
        
        # Reduce resources for retry attempts
        if attempt == 1:
            max_gaussians = 100000
            densify_interval = 200
        elif attempt == 2:
            max_gaussians = 50000
            densify_interval = 400
        
        return ChunkConfig(
            chunk_id=chunk_id,
            max_gaussians=max_gaussians,
            densify_interval=densify_interval,
            num_iters=num_iters,
            attempt=attempt
        )
    
    def train_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        Train a single chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Training results
        """
        chunk_dir = self.tasks_dir / chunk_id
        
        # Get current attempt
        chunk_status = self.db_manager.get_chunk_status(chunk_id)
        attempt = chunk_status['attempt'] if chunk_status else 0
        
        # Create configuration
        config = self.create_chunk_config(chunk_id, attempt)
        
        # Training configuration for worker
        worker_config = {
            'num_iterations': config.num_iters,
            'lr_positions': 1.6e-4,
            'lr_opacities': 1e-2,
            'lr_scales': 1e-3,
            'lr_rotations': 1e-3,
            'lr_colors': 5e-3,
            'densify_interval': config.densify_interval,
            'output_dir': str(self.output_dir)
        }
        
        try:
            # Mark as processing
            self.db_manager.update_chunk_status(chunk_id, 'PROCESSING')
            
            # Train chunk
            results = self.worker.train_chunk(str(chunk_dir), worker_config)
            
            if results['status'] == 'completed':
                self.db_manager.update_chunk_status(chunk_id, 'COMPLETED')
                self.logger.info(f"Chunk {chunk_id} completed: {results['num_gaussians']} Gaussians")
            else:
                raise Exception(results.get('error', 'Unknown error'))
            
            return results
            
        except ChunkOOMError as e:
            self.logger.warning(f"OOM in {chunk_id}, attempt {attempt}: {e}")
            
            if attempt < config.max_attempts - 1:
                # Retry with reduced resources
                new_config = self.create_chunk_config(chunk_id, attempt + 1)
                self.db_manager.increment_chunk_attempt(chunk_id, new_config)
                self.logger.info(f"Retrying {chunk_id} with attempt {attempt + 1}")
                return {'status': 'retry', 'chunk_id': chunk_id}
            else:
                # Max attempts reached
                self.db_manager.update_chunk_status(chunk_id, 'FAILED', str(e))
                self.logger.error(f"Chunk {chunk_id} failed after {attempt + 1} attempts")
                return {'status': 'failed', 'chunk_id': chunk_id, 'error': str(e)}
        
        except Exception as e:
            self.logger.error(f"Error training {chunk_id}: {e}")
            self.db_manager.update_chunk_status(chunk_id, 'FAILED', str(e))
            return {'status': 'failed', 'chunk_id': chunk_id, 'error': str(e)}
    
    def run_pipeline(self):
        """Run the complete training pipeline."""
        self.logger.info("Starting 3D Gaussian Splatting pipeline")
        
        # Initialize chunks
        self.initialize_chunks()
        
        # Get initial pending chunks
        pending_chunks = self.db_manager.get_pending_chunks()
        
        if not pending_chunks:
            self.logger.info("No pending chunks to process")
            return
        
        self.logger.info(f"Processing {len(pending_chunks)} chunks")
        
        # Process chunks sequentially (single GPU mode)
        total_chunks = len(self.manifest)
        completed_chunks = 0
        failed_chunks = 0
        
        start_time = time.time()
        
        while pending_chunks:
            # Get priority queue
            priority_queue = self.prioritizer.get_priority_queue(pending_chunks)
            
            if not priority_queue:
                break
            
            # Process highest priority chunk
            _, chunk_id = heapq.heappop(priority_queue)
            
            # Remove from pending list
            pending_chunks.remove(chunk_id)
            
            # Update progress
            completed_chunks += failed_chunks + 1
            progress = f"[{completed_chunks}/{total_chunks} chunks] {chunk_id}: "
            
            # Train chunk
            result = self.train_chunk(chunk_id)
            
            if result['status'] == 'completed':
                elapsed = time.time() - start_time
                progress += f"COMPLETED ({result['num_gaussians']} Gaussians, "
                progress += f"{result['num_iterations']} iters, {elapsed/60:.1f}m)"
                print(progress)
                
            elif result['status'] == 'retry':
                pending_chunks.append(chunk_id)  # Re-queue for retry
                
            elif result['status'] == 'failed':
                failed_chunks += 1
                elapsed = time.time() - start_time
                progress += f"FAILED ({result.get('error', 'Unknown error')}, {elapsed/60:.1f}m)"
                print(progress)
        
        # Final summary
        total_time = time.time() - start_time
        all_chunks = self.db_manager.get_all_chunks()
        
        completed = sum(1 for chunk in all_chunks if chunk['status'] == 'COMPLETED')
        failed = sum(1 for chunk in all_chunks if chunk['status'] == 'FAILED')
        
        print(f"\n=== Pipeline Summary ===")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time/60:.1f} minutes")
        
        self.logger.info(f"Pipeline completed: {completed} completed, {failed} failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Orchestrate 3DGS training pipeline")
    parser.add_argument("--tasks_dir", required=True, help="Directory containing chunk tasks")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        tasks_dir=args.tasks_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run pipeline
    orchestrator.run_pipeline()


if __name__ == "__main__":
    import torch  # Import here to avoid circular imports
    main()
