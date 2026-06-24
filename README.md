# 🚀 CARLA Grandmaster AI Suite v15.0

Selamat datang di Master Suite dari project Traffic Vision AI. Folder ini berisi seluruh evolusi kode dari PPO 1 hingga Grandmaster v15.0 — arsitektur terinspirasi Tesla/Waymo yang dirancang untuk melatih agen otonom tingkat tinggi di simulator CARLA.

---

## 📂 Struktur File Utama

| File | Fungsi |
|------|--------|
| `1_TRAIN_MASTER_v15.py` | Mesin pelatihan utama (Tesla/Waymo Architecture) |
| `1_COMPETITION_MASTER.py` | Multi-Agent Competition Training |
| `2_MONITOR_GRAPH.py` | Visualisasi real-time grafik Reward & Loss |
| `2_OBSERVER_AGENT.py` | Real-time training monitor & report generator |
| `2_BATTLE_ARENA.py` | Head-to-Head AI Battle & Tournament Bracket |
| `3_TEST_LIVE_v15.py` | Live evaluation AI (60 FPS, TensorRT support) |
| `3_COMPETITION_VIEWER.py` | Leaderboard viewer & agent analysis |
| `3_FINETUNE_MASTER.py` | Resume/fine-tune dari checkpoint tertentu |
| `4_COLLECT_HUMAN.py` | Rekam data mengemudi manusia (Imitation Learning) |
| `4_DEPLOY_MASTER_v15.py` | Export pipeline: ONNX → TensorRT FP16/INT8 |
| `gpu_optimizer.py` | ⚡ Auto-detect GPU & optimize (RTX 4080 tuned) |
| `START.bat` | Menu interaktif lengkap (12 opsi, auto-optimize GPU) |
| `START_COMPETITION.bat` | Menu interaktif khusus Competition |
| `core/` | Modul inti: SensorSuite, SafetyShield, CostFunction, dll. |

---

## ⚠️ Prasyarat: CARLA Server

Sebelum menjalankan script apapun, **CARLA Simulator harus sudah berjalan**:

```powershell
# Normal (dengan render):
CarlaUE4.exe

# Hemat RAM (Low Quality):
CarlaUE4.exe -quality-level=Low

# Training only (tanpa render, paling hemat):
CarlaUE4.exe -quality-level=Low -RenderOffScreen
```

Default koneksi: `localhost:2000`

---

## 🛠️ Cara Menjalankan — Semua Mode

Pastikan virtual environment aktif terlebih dahulu:
```powershell
cd D:\AI\traffic_vision
venv\Scripts\activate
```

---

### 🎮 Mode 1: MANUAL DRIVE (Nyetir Sendiri)

Untuk merekam data mengemudi manusia (Behavioral Cloning / Imitation Learning):

```powershell
python master_suite\4_COLLECT_HUMAN.py
```

**Kontrol Keyboard:**

| Tombol | Fungsi |
|--------|--------|
| `↑` UP | Gas |
| `↓` DOWN | Rem |
| `←` LEFT | Belok kiri |
| `→` RIGHT | Belok kanan |
| `S` | **Save data** demo & keluar |
| `ESC` | Cancel (keluar tanpa save) |

- Recording otomatis dimulai saat kamu mulai bergerak.
- Data disimpan ke `human_demos/observations.npy` dan `actions.npy`.
- Format: Dict of arrays (camera, lidar, state) — siap pakai untuk training BC.

---

### 🤖 Mode 2: AI LIVE TEST (Nonton AI Nyetir)

Melihat hasil AI menyetir secara real-time (60 FPS) dengan HUD lengkap:

```powershell
python master_suite\3_TEST_LIVE_v15.py
```

**Fitur:**
- Auto-load model terbaik (prioritas: TensorRT → ONNX → PyTorch SB3)
- HUD real-time: speed, steer, throttle, brake, distance, collisions
- Safety Shield status (intervensi otomatis anti-tabrakan)
- Backend indicator (PyTorch / ONNX / TensorRT)

**Kontrol Keyboard:**

| Tombol | Fungsi |
|--------|--------|
| `SPACE` (tahan) | Manual override — ambil alih dari AI |
| `←` `→` `↑` `↓` | Kontrol saat mode override |
| `ESC` | Keluar |

---

### 🏋️ Mode 3: TRAINING (Latih AI)

Mulai atau lanjutkan training Reinforcement Learning (PPO):

```powershell
python master_suite\1_TRAIN_MASTER_v15.py
```

Atau gunakan menu interaktif:
```powershell
master_suite\START.bat
# Pilih [1] TRAINING
```

**Fitur v15:**
- Tesla/Waymo inspired architecture
- Multi-channel CostFunction (progress, safety, comfort, rules)
- Route Planner (waypoint following, bukan random exploration)
- Safety Shield always-on
- Curriculum Learning (4 stages)
- Recall Memory (long-term spatial knowledge)
- Recovery Module (off-road self-rescue)
- ADR (Automatic Domain Randomization)
- Rival AI (Self-Play)

**Auto-resume** dari checkpoint terakhir jika ada.

---

### 🏆 Mode 4: COMPETITION (Multi-Agent Battle)

Jalankan kompetisi multi-agen:

```powershell
master_suite\START_COMPETITION.bat
```

**Menu:**
| Pilihan | Fungsi |
|---------|--------|
| `[1]` | Multi-Agent Training Competition |
| `[2]` | View Leaderboard |
| `[3]` | Live Dashboard (Real-time) |
| `[4]` | Head-to-Head Battle (1v1) |
| `[5]` | Championship Bracket |
| `[6]` | View Agent Details |
| `[7]` | View Battle History |
| `[8]` | View Memory Analysis |

---

### 📊 Mode 5: MONITOR (Pantau Training)

**TensorBoard:**
```powershell
tensorboard --logdir master_suite\tb_logs --port 6006
```
Buka `http://localhost:6006` di browser.

**Observer Agent (terminal):**
```powershell
# Real-time watch:
python master_suite\2_OBSERVER_AGENT.py --mode watch

# Generate report:
python master_suite\2_OBSERVER_AGENT.py --mode report
```

**Grafik Matplotlib:**
```powershell
python master_suite\2_MONITOR_GRAPH.py
```

---

### 🚀 Mode 6: DEPLOY (Export Model)

Export model untuk production deployment:

```powershell
# Full pipeline: Validate → ONNX → TensorRT → Benchmark
python master_suite\4_DEPLOY_MASTER_v15.py --action all

# Hanya validasi model:
python master_suite\4_DEPLOY_MASTER_v15.py --action validate --episodes 10

# Hanya export ONNX + TensorRT:
python master_suite\4_DEPLOY_MASTER_v15.py --action export

# Hanya benchmark kecepatan inference:
python master_suite\4_DEPLOY_MASTER_v15.py --action benchmark
```

Output disimpan di folder `deploy/`.

---

## 📜 Arsitektur v15.0

```
┌─────────────────────────────────────────────────────────────┐
│                 TRAINING PIPELINE v15                        │
│                                                              │
│  Sensors ──▶ Feature Extractor ──▶ PPO ──▶ Action            │
│   │              │                    │       │              │
│   │     LiDAR Spatial Encoder    Cost Function │              │
│   │     CNN Semantic Context     (4 channels)  │              │
│   │     State Encoder            Progress      │              │
│   │         │                    Safety    ◀───┘              │
│   │    Fusion MLP               Comfort                      │
│   │                             Rules                        │
│   │                                                          │
│   └──▶ Safety Shield ──▶ CARLA ──▶ Sensors (loop)           │
│                                                              │
│  + Route Planner (Tesla-style waypoint following)            │
│  + Curriculum Learning (4 stages)                            │
│  + Recall Memory (long-term spatial knowledge)               │
│  + Recovery Module (off-road self-rescue)                    │
│  + ADR (Automatic Domain Randomization)                      │
│  + TensorRT export pipeline                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Original Vision (The 8 Core Pillars)

*Poin-poin landasan utama project sejak awal:*

1. **Simulation and Environment Setup**: Simulasi dunia nyata yang akurat dengan sensor Vision, Radar, dan LiDAR.
2. **Phased Learning (Curriculum)**: Belajar bertahap dari jalan lurus hingga interaksi kompleks.
3. **Supervised Learning (Human-Teaching)**: Mode manual untuk merekam gaya mengemudi manusia sebagai dataset.
4. **Autonomous Reinforcement Learning**: AI belajar mandiri melalui sistem reward (hadiah) dan penalti (hukuman).
5. **Hybrid Integration**: Menggabungkan demonstrasi manusia dengan eksplorasi mandiri AI.
6. **Transition to Real-World**: Model dirancang untuk siap ditransfer ke sensor fisik di dunia nyata.
7. **Human-in-the-Loop Control**: Fitur manual override kapan saja untuk koreksi perilaku AI.
8. **Safety and Robustness Measures**: Melatih AI di skenario ekstrem (cuaca buruk & rintangan mendadak).

---

## ⚙️ Technical Specification

### Sensor Suite
- **Primary Vision**: Semantic Segmentation Camera (80×60)
- **LiDAR**: 16-sector spatial encoding, 32 channels, max range 50m
- **State Vector**: 28-dimensional (speed, steer, traffic, route, safety, etc.)
- **Collision & Lane Invasion**: Event-based detection

### Model
- **Architecture**: PPO with custom `GrandmasterExtractor` (CNN + LiDAR + State fusion)
- **Hyperparameters**: `lr=3e-4`, `n_steps=4096`, `batch_size=512`, `ent_coef=0.015`
- **Action Space**: Continuous `[steer, throttle, brake]` → `[-1,1] × [-1,1] × [0,1]`
- **Feature Dim**: 256 → MLP `[256, 128]` (pi) + `[256, 128]` (vf)

### Safety
- **Safety Shield**: Always-on collision avoidance (LiDAR-based)
- **Recovery Module**: Auto self-rescue saat off-road
- **ADR**: Automatic Domain Randomization (weather, lighting)

### Deploy
- **ONNX Runtime**: Cross-platform inference (GPU + CPU fallback)
- **TensorRT**: FP16/INT8 quantization untuk edge deployment
- **Benchmark**: Membandingkan PyTorch vs ONNX vs TensorRT

---

## ⚡ RTX 4080 Laptop Optimizations

Semua optimisasi berikut **otomatis aktif** saat menjalankan START.bat atau script manapun:

### Auto-Applied GPU Settings

| Optimization | Flag | Speedup |
|-------------|------|---------|
| **TF32 Math** | `NVIDIA_TF32_OVERRIDE=1` | ~2-3x matmul |
| **cuDNN Benchmark** | `CUDNN_BENCHMARK=1` | Auto-fastest conv |
| **cuDNN v8 Flash** | `TORCH_CUDNN_V8_API_ENABLED=1` | Reduced latency |
| **CUDA Memory Pool** | `expandable_segments:True` | Less fragmentation |
| **Max Split Size** | `max_split_size_mb:256` | Prevents OOM |
| **Device Pinning** | `CUDA_VISIBLE_DEVICES=0` | No GPU switching |

### TensorRT Pipeline (RTX 40xx Optimal)

```
PyTorch Model → ONNX (opset 17) → TensorRT FP16 → 2GB Workspace
                                 → TensorRT INT8 (optional)
```

- **FP16**: Best untuk RTX 4080 (Ada Lovelace native FP16)
- **INT8**: Lebih cepat tapi butuh calibration data
- **Workspace**: 2 GB (optimal untuk 12GB VRAM)

### Cara Kerja

1. `START.bat` → set env vars CUDA/TF32/cuDNN
2. `gpu_optimizer.py` → detect GPU, apply fine-tuning, warm up CUDA
3. `core/constants.py` → `apply_gpu_opts()` dipanggil oleh setiap script
4. PyTorch `th.backends.cuda.matmul.allow_tf32 = True` di setiap entry point
5. TensorRT auto-export setelah training selesai

### GPU Optimizer Commands

```powershell
# Full system check (GPU + RAM + TensorRT status):
python gpu_optimizer.py --mode check

# Apply training optimizations:
python gpu_optimizer.py --mode optimize --training

# Auto-build TensorRT engines:
python gpu_optimizer.py --mode export

# Kill bloatware + free RAM:
python gpu_optimizer.py --mode cleanup
```

---

## 💡 Tips Performa

| Masalah | Solusi |
|---------|--------|
| RAM tinggi (>80%) | `START.bat → [10] Cleanup` atau `python gpu_optimizer.py --mode cleanup` |
| Training tanpa render | CARLA: `-quality-level=Low -RenderOffScreen` |
| Disk penuh | Hapus checkpoint lama di `checkpoints/` (177 files × ~12MB = ~2.1GB) |
| GPU OOM | Kurangi `batch_size` di `1_TRAIN_MASTER_v15.py` (default 512) |
| CARLA timeout | Restart CARLA, pastikan port 2000 tidak conflict |
| Inference lambat | Run `START.bat → [6] Deploy` untuk build TensorRT FP16 |
| VRAM leak | CARLA: restart server setiap ~50 episode |

---

*Project ini dikembangkan secara inkremental dengan membuang fitur cacat dan memperkuat fitur juara dari setiap iterasi PPO (v1 → v15). Optimized for NVIDIA RTX 4080 Laptop (Ada Lovelace).*

