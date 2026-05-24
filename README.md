# 🌌 Gaussian Splatting Pipeline (RTX 3050 Optimized)

A high-performance 3D Gaussian Splatting implementation specifically engineered to run within the **4GB VRAM** constraints of the **RTX 3050 (Laptop/Desktop)**. This pipeline bridges **WSL2** and **Windows** for a seamless training and visualization experience.

---

## 🛠️ Implementation Summary

Currently, we have established a robust end-to-end pipeline covering environment setup, data validation, and optimized training.

### 1. Robust Environment (`setup.sh`)
- **Automated Provisioning**: One-script setup for Miniconda, Python 3.10, and CUDA 12.1.
- **Isolated Stack**: Custom `gaussian-splat` conda environment with `gsplat 1.3.0` and `torch 2.2.0`.
- **System Integration**: Automated installation of COLMAP via `apt` within WSL2.

### 2. Hardware-Aware Validation (`smoke_test.py`)
- **GPU Verify**: Real-time CUDA availability check.
- **Hardware Lock**: Specifically validates against the **RTX 3050** to ensure 4GB memory-safe parameters are used.

### 3. Smart COLMAP Parser (`validate_colmap.py`)
- **Binary Recovery**: Custom robust parser for COLMAP `sparse/0` binaries.
- **Error Resilience**: Handles special characters in filenames (UTF-8 recovery) and prevents binary offset drift.
- **Visual Assurance**: Integrated Open3D point cloud visualization for data quality checks.

### 4. VRAM-Optimized Trainer (`train_single_node.py`)
- **Real-Data Initialization**: Successfully integrated **8,051 sparse points** from COLMAP for high-fidelity startup.
- **4GB VRAM Guard**: Implemented `max_gaussians` (120k) and image downscaling to prevent OOM errors.
- **Pro-Level Export**: Generates industry-standard `.ply` files compatible with **SuperSplat**, **Polycam**, and **Splat AI** by utilizing Spherical Harmonics (`f_dc`) encoding.

---

## 📂 Project Structure

```text
├── .env                  # Config for Supabase/Anedya modules
├── EXECUTION_GUIDE.md   # Step-by-step run instructions
├── setup.sh             # Linux environment setup script
├── smoke_test.py        # Hardware/Version validator
├── validate_colmap.py   # Dataset & Reconstruction validator
├── train_single_node.py  # Core training & export engine
└── requirements.txt     # Python dependency manifest
```

---

## 🚀 Quick Start

1.  **Prepare WSL2 Environment**:
    ```bash
    ./setup.sh
    ```
2.  **Verify Data**:
    ```bash
    python validate_colmap.py --data_dir data
    ```
3.  **Run Optimized Training**:
    ```bash
    python train_single_node.py --data_dir data --iterations 30000
    ```

---

## 🔗 Integration with Other Modules

The pipeline is pre-configured to sync with:
- **Supabase**: Remote model storage and metadata tracking.
- **Anedya**: Real-time telemetry during training (mock-ready).
- **SuperSplat**: Optimized for the PlayCanvas Gaussian Splatting editor.

---

> [!NOTE]
> This project is currently in the **Refinement Phase**. All core systems are operational and verified on local hardware.
