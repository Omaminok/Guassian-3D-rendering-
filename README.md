# Splat-Grid — Distributed 3D Gaussian Splatting Engine

Splat-Grid partitions a large 3D Gaussian Splatting (3DGS) scene across multiple consumer machines. Each machine trains only a spatial sub-volume (*voxel*), preventing Out-of-Memory errors while allowing parallel reconstruction.

The system is a **pure TCP, 6-node architecture** — no HTTP servers, no cloud services, no CUDA toolkit required on workers (uses a built-in pure-PyTorch rasterizer).

---

## Table of Contents

1. [How It Works — Big Picture](#how-it-works)
2. [Node Reference](#node-reference)
   - [State Manager](#1-state-manager--state_managerserverpy-port-5001)
   - [Data Engine](#2-data-engine--data_engineserverpy-port-5002)
   - [Stitcher](#3-stitcher--stitcherserverpy-port-5003)
   - [Worker](#4-worker--workerclientpy)
   - [Watchdog](#5-watchdog--watchdogclientpy)
   - [Master](#6-master--masterclientpy)
3. [Shared Infrastructure](#shared-infrastructure)
   - [TCP Protocol](#tcp-protocol-sharedSocket_utilspy)
   - [Soft Rasterizer](#soft-rasterizer-splat_gridsoft_rasterizerpy)
   - [COLMAP Utilities](#colmap-utilities-splat_gridcolmap_utilspy)
   - [PLY Utilities](#ply-utilities-splat_gridply_utilspy)
4. [Directory Structure](#directory-structure)
5. [Environment Configuration](#environment-configuration)
6. [Setup & Installation](#setup--installation)
7. [Running the Swarm](#running-the-swarm)
8. [Memory Safety Guarantees](#memory-safety-guarantees)
9. [Data Flow Diagram](#data-flow-diagram)

---

## How It Works

```
Input: COLMAP dataset (cameras.bin, images.bin, points3D.bin, images/)
         │
         ▼
  [Master] triggers INITIALIZE on Data Engine
         │
         ▼
  [Data Engine] partitions scene into N³ voxel tasks, pushes them to State Manager
         │
         ▼
  [State Manager] holds all tasks in SQLite (pending → in_progress → completed)
         │
  ┌──────┴──────┐
  │   Workers   │  (run on any machine on the LAN)
  │  pull tasks │  download chunk ZIP from Data Engine
  │  train      │  750 iterations of 3DGS per voxel
  │  upload PLY │  send result to Stitcher
  └──────┬──────┘
         │
  [Stitcher] accumulates chunk PLY files
         │
  [Master] detects 100% completion, triggers STITCH_ALL
         │
         ▼
  Output: stitched_output.ply  (3DGS-compatible, opens in SuperSplat / SIBR viewer)
```

Between each pair of steps, **Watchdog** runs in the background — every 30 s it scans for stale tasks (worker died mid-job) and resets them to `pending` so another worker can pick them up.

---

## Node Reference

### 1. State Manager — `state_manager/server.py` (port 5001)

**Role:** Central task registry. The single source of truth for what work needs to be done, what is in progress, and what is finished. All other nodes talk to this server.

**Technology:** Python `socket` server + SQLite database. One thread per connection. Thread-local SQLite connections avoid locking issues.

**Database schema (`tasks` table):**

| Column | Type | Description |
|---|---|---|
| `task_id` | TEXT PK | e.g. `voxel_0_1_2` |
| `status` | TEXT | `pending` / `in_progress` / `completed` |
| `bbox_min` | TEXT | JSON `[x, y, z]` — voxel lower bound |
| `bbox_max` | TEXT | JSON `[x, y, z]` — voxel upper bound |
| `image_names` | TEXT | JSON list of image filenames for this chunk |
| `n_points` | INTEGER | Sparse COLMAP points inside this voxel |
| `worker_id` | TEXT | ID of the worker currently processing it |
| `last_heartbeat` | REAL | Unix timestamp of last heartbeat from worker |
| `created_at` | REAL | Unix timestamp |

**TCP commands accepted:**

| Command | Payload | Response |
|---|---|---|
| `CREATE_TASKS` | `{"tasks": [...]}` | `{"status":"ok","created":N}` |
| `CLAIM_TASK` | `{"worker_id":"..."}` | `{"status":"ok","task":{...}}` or `{"status":"none"}` |
| `HEARTBEAT` | `{"task_id":"...","worker_id":"..."}` | `{"status":"ok"}` |
| `COMPLETE_TASK` | `{"task_id":"...","worker_id":"..."}` | `{"status":"ok"}` |
| `RESET_TASK` | `{"task_id":"..."}` | `{"status":"ok"}` |
| `GET_STATUS` | *(no extra fields)* | `{"total":N,"pending":N,"in_progress":N,"completed":N}` |
| `GET_STALE` | `{"threshold_seconds":60}` | `{"stale":["task_id",...]}` |

**Key design decisions:**
- Sends exactly **one response per connection** (except State Manager which keeps the connection open for multiple commands per client session).
- SQLite `INSERT OR IGNORE` prevents duplicate tasks if Master is re-run.
- `CLAIM_TASK` uses `SELECT ... LIMIT 1` FIFO ordering — tasks are always assigned oldest-first.

---

### 2. Data Engine — `data_engine/server.py` (port 5002)

**Role:** Reads the raw COLMAP dataset, partitions the scene into 3-D voxel sub-tasks, and serves per-chunk data (COLMAP binaries + images) as ZIP archives to workers on demand.

**Technology:** Python `socket` server. One thread per connection. In-memory task store (dict) populated on `INITIALIZE`.

**TCP commands accepted:**

| Command | Payload | Response |
|---|---|---|
| `INITIALIZE` | `{"data_dir":"/path/to/colmap","grid":N}` | `{"status":"ok","n_tasks":N}` |
| `DOWNLOAD_CHUNK` | `{"task_id":"voxel_x_y_z"}` | `{"status":"ok"}` then raw ZIP bytes (length-prefixed) |
| `GET_TASK_META` | `{"task_id":"..."}` | `{"bbox_min":[...],"bbox_max":[...],"n_points":N}` |

**Scene partitioning algorithm:**
1. Load `points3D.bin`, compute global axis-aligned bounding box with a 2% margin.
2. Divide the bbox into `grid × grid × grid` equal cells.
3. For each 3-D point, determine which cell it falls in.
4. Create one task per **non-empty** cell (cells with zero sparse points are skipped).
5. All images are assigned to every task (workers filter by camera overlap themselves).
6. Push the task list to State Manager via `CREATE_TASKS`.

**ZIP bundle contents (per chunk):**
```
voxel_x_y_z.zip
├── sparse/0/cameras.bin
├── sparse/0/images.bin
├── sparse/0/points3D.bin
└── images/<all image filenames>
```

Workers unzip this locally and have a self-contained COLMAP dataset to train on.

---

### 3. Stitcher — `stitcher/server.py` (port 5003)

**Role:** Receives finished `.ply` chunk files from workers and, when triggered, merges them all into a single 3DGS-compatible output PLY.

**Technology:** Python `socket` server. One thread per connection. Thread-safe upload lock. Chunk files persisted to `stitcher/chunks/` directory.

**TCP commands accepted:**

| Command | Payload | Response |
|---|---|---|
| `UPLOAD_RESULT` | `{"task_id":"voxel_x_y_z"}` | `{"status":"ok"}` then receives raw PLY bytes |
| `STITCH_ALL` | `{"output_path":"stitched_output.ply"}` | `{"status":"ok","total_gaussians":N,"output":"..."}` |
| `GET_CHUNKS` | *(no extra fields)* | `{"chunks":["task_id",...]}` |

**Upload protocol (two-phase):**
1. Worker sends `UPLOAD_RESULT` message.
2. Stitcher acknowledges with `{"status":"ok"}`.
3. Worker streams the raw PLY file (4-byte length header + bytes).
4. Stitcher writes it to `stitcher/chunks/<task_id>.ply`.

**Stitching (`STITCH_ALL`):**
- Uses `splat_grid.ply_utils.stitch_ply_files()`.
- Reads the PLY header from the first chunk as a template.
- Concatenates all binary vertex data sections directly (no re-encoding).
- Rewrites the `element vertex N` count to the total sum.
- Extremely fast and RAM-efficient — only headers are parsed; data is streamed.

> **Crash recovery:** On restart the Stitcher re-scans `stitcher/chunks/*.ply` for any files that arrived before a restart, so no chunks are lost.

---

### 4. Worker — `worker/client.py`

**Role:** The GPU/CPU training node. Pulls tasks from State Manager, downloads the chunk ZIP from Data Engine, runs the 3DGS training loop, and uploads the resulting `.ply` to Stitcher.

**Technology:** Pure Python + PyTorch. No CUDA toolkit or `gsplat` CUDA extension required — uses the built-in **soft rasterizer** (`splat_grid/soft_rasterizer.py`).

**Full task lifecycle:**
```
1. CLAIM_TASK  →  State Manager
2. DOWNLOAD_CHUNK  →  Data Engine  (receives ZIP, unzips locally)
3. Start heartbeat thread  (sends HEARTBEAT every 20 s)
4. train_chunk()  —  3DGS training loop (750 iterations)
5. write_ply()  —  export chunk to .ply
6. UPLOAD_RESULT  →  Stitcher  (streams .ply)
7. COMPLETE_TASK  →  State Manager
8. Cleanup temp files
9. Loop back to step 1
```

**Training loop (`train_chunk`):**
- Loads cameras, image poses, and sparse points from the unzipped COLMAP data.
- Initialises Gaussians from sparse points inside the voxel bbox (or 2 000 random Gaussians if the voxel is empty).
- Runs 750 Adam steps, rendering one random training image per step.
- Computes L1 loss + optional SSIM loss.
- Runs **densification** (splitting high-gradient Gaussians) and **pruning** (removing transparent/out-of-bbox Gaussians) every 100 steps between steps 500–750.
- Exports the final Gaussian parameters to a binary PLY file.

**Memory safety constants (DO NOT relax):**

| Parameter | Value | Reason |
|---|---|---|
| `SH_DEGREE` | 0 | 1 SH coefficient per channel — minimum memory |
| `MAX_GAUSSIANS` | 50 000 | Hard cap per chunk |
| `downscale` | 4 | ¼ resolution → ~16× fewer pixels |
| `iterations` | 750 | Low enough to prevent OOM on 4 GB VRAM |
| Cache clear | every 50 steps | Prevents CUDA fragment accumulation |

**Heartbeat thread:** A background `threading.Thread` sends a `HEARTBEAT` to State Manager every 20 seconds. If the worker crashes or hangs, Watchdog detects the missed heartbeat and re-queues the task.

**CLI arguments:**
```
python worker/client.py
  [--worker_id  "my-laptop"]   # auto-generated UUID prefix if omitted
  [--work_dir   "./tmp"]        # temp directory (system temp if omitted)
  [--iterations 750]
  [--downscale  4]
  [--max_tasks  0]              # 0 = run until queue empty
```

---

### 5. Watchdog — `watchdog/client.py`

**Role:** Fault-tolerance daemon. Runs in the background and automatically recovers from worker failures by resetting stale tasks.

**Technology:** Pure Python polling loop. No server port — purely a TCP client.

**Algorithm (runs every 30 seconds):**
1. `GET_STATUS` → State Manager: print progress summary.
2. If all tasks are `completed`, exit cleanly.
3. `GET_STALE` → State Manager: ask for `in_progress` tasks whose `last_heartbeat` is older than 60 seconds.
4. For each stale task: `RESET_TASK` → State Manager → task reverts to `pending`.

**Why this matters:** A worker can die (OOM, network drop, laptop sleep) mid-training. Without Watchdog, that task stays `in_progress` forever and the pipeline never completes. Watchdog guarantees *at-least-once execution* for every voxel task.

**CLI arguments:**
```
python watchdog/client.py
  [--interval  30]   # seconds between polls
  [--threshold 60]   # seconds of heartbeat silence = "stale"
```

---

### 6. Master — `master/client.py`

**Role:** The CLI orchestrator. Kicks off the pipeline, monitors progress, and triggers final stitching when all tasks complete.

**Technology:** Pure Python. Runs in the foreground. No server port — purely a TCP client.

**Execution flow:**
```
1. Validate --data_dir exists locally
2. INITIALIZE  →  Data Engine  (triggers partitioning + task creation)
3. Poll GET_STATUS  →  State Manager  every 5 s  (prints progress table)
4. When completed == total: STITCH_ALL  →  Stitcher
5. Print final output path and exit
```

**CLI arguments:**
```
python master/client.py
  --data_dir  .\data           # path to COLMAP dataset (required)
  [--output   stitched_output.ply]
  [--grid     2]               # scene is split into grid³ voxels (default 2 → 8 cells)
```

> **Grid sizing guide:** `--grid 1` = 1 task (test/debug), `--grid 2` = up to 8 tasks, `--grid 4` = up to 64 tasks. Only non-empty voxels become actual tasks — a `--grid 4` run with a small scene may create only ~30 tasks (as seen in live output).

---

## Shared Infrastructure

### TCP Protocol — `shared/socket_utils.py`

All inter-node communication uses a **4-byte length-prefix framing protocol** over raw TCP:

```
┌──── 4 bytes (big-endian uint32) ────┬──── N bytes ────────────┐
│  Payload length N                   │  UTF-8 JSON string      │
└────────────────────────────────────┴─────────────────────────┘
```

Binary file transfers (ZIPs, PLYs) use the same framing but with raw bytes instead of JSON.

**Helpers:**

| Function | Description |
|---|---|
| `send_msg(sock, dict)` | Serialize dict → JSON → send with 4-byte header |
| `recv_msg(sock)` | Receive length-prefixed message → parse JSON → return dict |
| `send_file(sock, path)` | Stream a binary file with 4-byte length header |
| `recv_file(sock, dest)` | Receive a streamed binary file and write to disk |
| `connect_with_retry(host, port)` | Connect with up to 12 retries (5 s delay) — handles boot-order races |

---

### Soft Rasterizer — `splat_grid/soft_rasterizer.py`

A **pure-PyTorch differentiable Gaussian rasterizer** that replaces `gsplat`'s CUDA extension. No `CUDA_HOME`, no `cl.exe`, no JIT compilation required.

**Algorithm per frame:**
1. Build 3-D covariance matrices: `Σ3D = R · diag(s²) · Rᵀ` (from quaternions + log-scales).
2. Project each Gaussian to camera space via the view matrix.
3. Compute 2-D covariance via the Jacobian of perspective projection: `Σ2D = J · Rcam · Σ3D · Rcam' · Jᵀ`.
4. Invert `Σ2D` for per-pixel Mahalanobis distance computation.
5. Sort Gaussians front-to-back by depth.
6. For each Gaussian, compute a 2-D Gaussian blob over its ±3σ bounding box using `F.pad` (fully out-of-place).
7. Alpha-composite all contributions: `colour = Σᵢ Tᵢ · αᵢ · cᵢ` where `Tᵢ = ∏_{j<i} (1 − αⱼ)`.

Gradients flow through all parameters: `means`, `quats`, `scales`, `opacities`, `sh_colors`. The Adam optimiser and densification/pruning logic are **mathematically identical** to a gsplat-based pipeline.

---

### COLMAP Utilities — `splat_grid/colmap_utils.py`

Binary parsers for the COLMAP sparse reconstruction format:

| Function | File | Output |
|---|---|---|
| `read_cameras_binary(path)` | `cameras.bin` | `{cam_id: {fx, fy, cx, cy, width, height}}` |
| `read_images_binary(path)` | `images.bin` | `{img_id: {name, cam_id, R[3×3], t[3]}}` |
| `read_points3D_binary(path)` | `points3D.bin` | `{pt_id: {xyz[3], rgb[3]}}` |
| `load_colmap_data(data_dir)` | all three | `(cameras, images, points3D)` |
| `load_images_for_worker(...)` | JPEG/PNG files | `{img_id: Tensor[H,W,3]}, {img_id: (H,W)}` |

Supports all COLMAP camera models: `SIMPLE_PINHOLE`, `PINHOLE`, `RADIAL`, `OPENCV`, `FISHEYE`, etc.

---

### PLY Utilities — `splat_grid/ply_utils.py`

| Function | Used by | Description |
|---|---|---|
| `write_ply(path, means, sh_colors, opacities, scales, quats)` | Worker | Exports Gaussian parameters to 3DGS-compatible binary PLY |
| `stitch_ply_files(input_paths, output_path)` | Stitcher | Concatenates chunk PLYs into one file — O(n) I/O, O(1) RAM |

The output PLY is compatible with:
- **SuperSplat** (web viewer)
- **INRIA SIBR viewer**
- **Postshot / Luma AI** (import)

---

## Directory Structure

```
gaussian-splatting/
│
├── master/
│   └── client.py          # Orchestrator CLI — runs last, in foreground
│
├── state_manager/
│   ├── server.py          # TCP server, port 5001 — SQLite task registry
│   └── tasks.db           # Auto-created SQLite database (gitignored)
│
├── data_engine/
│   └── server.py          # TCP server, port 5002 — COLMAP ZIP server
│
├── stitcher/
│   ├── server.py          # TCP server, port 5003 — PLY collector/stitcher
│   └── chunks/            # Auto-created — received .ply chunks (gitignored)
│
├── worker/
│   ├── client.py          # Training node — GPU/CPU, runs one per machine
│   └── .env               # Worker-side host/port config
│
├── watchdog/
│   ├── client.py          # Fault-tolerance daemon — reset stale tasks
│   └── .env               # Watchdog host/port config
│
├── shared/
│   └── socket_utils.py    # TCP framing + file transfer helpers
│
├── splat_grid/
│   ├── colmap_utils.py    # Binary COLMAP parsers
│   ├── ply_utils.py       # PLY writer + stitcher
│   └── soft_rasterizer.py # Pure-PyTorch differentiable Gaussian rasterizer
│
├── data/                  # Input COLMAP dataset (gitignored)
│   ├── sparse/0/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── images/
│
├── .env.local             # Shared IP/port config (all nodes read this)
├── start_swarm.ps1        # PowerShell: boots all 6 nodes locally
├── requirements.txt       # Python dependencies
└── stitched_output.ply    # Final output (gitignored)
```

---

## Environment Configuration

All nodes read their connection info from environment variables. For a local run, create/edit `.env.local` in the project root:

```ini
# .env.local
STATE_MANAGER_IP=127.0.0.1
STATE_MANAGER_PORT=5001

DATA_ENGINE_IP=127.0.0.1
DATA_ENGINE_PORT=5002

STITCHER_IP=127.0.0.1
STITCHER_PORT=5003
```

**For distributed runs across machines:**
- Replace `127.0.0.1` with the LAN IP of the machine running each server.
- State Manager, Data Engine, and Stitcher should run on the same machine (or at least have stable IPs).
- Workers and Watchdog only need the above three IPs and will work from any machine on the network.
- Run `allow_firewall.ps1` on the server machine to open the firewall ports.

---

## Setup & Installation

### Requirements

- Python 3.10+
- PyTorch 2.x (CPU is sufficient; CUDA optional for faster training)
- No CUDA Toolkit / MSVC required

### Install

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow python-dotenv pytorch-msssim

# Optional: faster training on NVIDIA GPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** `gsplat` is listed in `requirements.txt` but is **not required**. The system uses the built-in pure-PyTorch rasterizer. You may remove `gsplat` and `open3d` from `requirements.txt` if you don't need them.

### Prepare your COLMAP dataset

```
data/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── images/
    ├── frame001.jpg
    ├── frame002.jpg
    └── ...
```

Use `python convert.py` or standard COLMAP to generate the sparse reconstruction.

Validate your dataset:
```powershell
python validate_colmap.py --data_dir .\data
# Expected: ≥ 40 cameras, ≥ 500 sparse points
```

---

## Running the Swarm

### Option A — All-in-one (local, PowerShell)

```powershell
cd D:\Development\Try\gaussian-splatting
.\.venv\Scripts\Activate.ps1

# Boots all 6 nodes; Master runs in foreground
.\start_swarm.ps1 -DataDir .\data -Grid 2
```

Each node opens in its own PowerShell window. Master blocks until complete.

### Option B — Manual (one terminal per node)

Boot in this exact order:

```powershell
# Terminal 1 — State Manager (must be first)
.venv\Scripts\python.exe state_manager\server.py

# Terminal 2 — Stitcher
.venv\Scripts\python.exe stitcher\server.py

# Terminal 3 — Data Engine
.venv\Scripts\python.exe data_engine\server.py

# Terminal 4 — Watchdog (fault tolerance)
.venv\Scripts\python.exe watchdog\client.py

# Terminal 5 — Worker (one per machine; run as many as you want)
.venv\Scripts\python.exe worker\client.py --worker_id my-laptop

# Terminal 6 — Master (run last; blocks until done)
.venv\Scripts\python.exe master\client.py --data_dir .\data --grid 2
```

### Option C — Distributed across multiple machines

On the **server machine** (runs the three TCP servers):
```powershell
.venv\Scripts\python.exe state_manager\server.py
.venv\Scripts\python.exe stitcher\server.py
.venv\Scripts\python.exe data_engine\server.py
```

On each **worker machine**, set the server's LAN IP in `.env.local`:
```ini
STATE_MANAGER_IP=192.168.1.100
DATA_ENGINE_IP=192.168.1.100
STITCHER_IP=192.168.1.100
```
Then:
```powershell
.venv\Scripts\python.exe worker\client.py --worker_id laptop-2
```

On the **master machine** (same `.env.local` as workers):
```powershell
.venv\Scripts\python.exe master\client.py --data_dir .\data --grid 4
```

---

## Memory Safety Guarantees

These constraints are locked in `worker/client.py` and must not be relaxed without testing on your weakest target machine:

| Constraint | Value | Effect |
|---|---|---|
| SH degree | 0 | 1 colour coefficient per Gaussian — minimum VRAM |
| Max Gaussians | 50 000 / chunk | Prevents unbounded densification memory growth |
| Image downscale | 4× | ¼ resolution → 16× fewer pixels per render |
| Max iterations | 750 | Low enough for 4 GB VRAM machines |
| `empty_cache()` | every 50 steps | Prevents CUDA memory fragmentation |
| Scale clamp | `max(scale) < 0.5` | Prunes elongated Gaussians that cause OOM |
| Bbox pruning | every densify step | Discards Gaussians that escaped the voxel |

---

## Data Flow Diagram

```
                     ┌─────────────────────────────────────────┐
                     │           state_manager :5001           │
                     │         (SQLite task registry)          │
                     └───┬──────────────┬──────────────────────┘
                         │              │
              CREATE_TASKS│        CLAIM_TASK / HEARTBEAT /
                         │        COMPLETE_TASK / GET_STATUS
                         │              │
          ┌──────────────┘    ┌─────────┴─────────┐
          │                   │                   │
┌─────────▼─────────┐  ┌──────▼──────┐   ┌───────▼──────┐
│   data_engine      │  │   worker    │   │   watchdog   │
│      :5002         │  │  (client)   │   │   (client)   │
│                    │  │             │   │              │
│  INITIALIZE ←──────┼──┤ master      │   │  GET_STALE   │
│  stores tasks      │  │  triggers   │   │  RESET_TASK  │
│                    │  │             │   │  every 30s   │
│  DOWNLOAD_CHUNK ◄──┼──┤ worker      │   └──────────────┘
│  streams ZIP       │  │  downloads  │
└────────────────────┘  │             │
                        │  trains     │
                        │  750 steps  │
                        │             │
              UPLOAD_RESULT│
                        │──►──────────────────────────────────┐
                        │                                     │
                        └─────────────────────────────────────►  stitcher :5003
                                                              │  (stores chunks)
                                                              │
                                                              │  ◄── STITCH_ALL
                                                              │      from master
                                                              │
                                                              ▼
                                                   stitched_output.ply
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefusedError` on worker start | Server nodes not running yet | Boot in order: State Manager → Stitcher → Data Engine → Worker |
| `INITIALIZE failed: data_dir not found` | Master's `--data_dir` path doesn't exist | Use an absolute path or ensure `.\data` is correct relative to CWD |
| Tasks stuck `in_progress` forever | Worker crashed without completing | Watchdog will reset them after 60 s; or run `RESET_TASK` manually |
| `No images loaded` in worker log | Images missing from data dir or not in COLMAP | Verify `data/images/` contains all files listed in `images.bin` |
| Progress shows `0/N completed` for long time | Only 1 worker running; training is slow on CPU | Add more workers; CPU training takes ~10 min/chunk at ¼ resolution |
| PLY file is empty or tiny | Worker completed but Stitcher crashed | Re-run Stitcher; it will find existing chunks in `stitcher/chunks/` |
