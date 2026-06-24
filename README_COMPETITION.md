# CARLA AI COMPETITION LEAGUE

Sistem **kompetisi multi-agent** untuk autonomous driving dengan fitur:

## 🎯 Fitur Utama

### 1. No-Reset Penalty System
- Agent **TIDAK reset** saat kena punishment
- Episode terus berjalan dengan penalty akumulatif
- Agent harus memutuskan: lanjut dengan penalty atau cari jalan recovery

### 2. Decision Making Module
- Agent "mikir 2x" sebelum mengambil keputusan berbahaya
- Otomatis mengurangi kecepatan saat:
  - Masuk area high danger (dari memory)
  - Akan keluar dari aspal
  - Ada obstacle di dekat

### 3. Recall Memory System
Setiap agent mengingat:
- 📍 **Collision Hotspots** - Lokasi sering terjadi tabrakan
- 🛣️ **Off-road Zones** - Area berbahaya untuk dihindari
- 🚦 **Red Light Zones** - Lokasi traffic light
- ✅ **Successful Route** - Rute terbaik yang pernah dilalui

### 4. Competitive Leaderboard
- Skor komposit: **Distance - Penalties - Safety Reduction**
- Real-time ranking
- Historical battle records

## 📁 Struktur File

```
master_suite/
├── 1_COMPETITION_MASTER.py   # Training multi-agent
├── 2_BATTLE_ARENA.py         # Head-to-head battle
├── 3_COMPETITION_VIEWER.py   # Leaderboard & analytics
├── START_COMPETITION.bat     # Menu launcher
├── checkpoints/              # Model checkpoints
├── agent_memories/           # Memory files per agent
└── leaderboard/
    └── leaderboard.json      # Database kompetisi
```

## 🚀 Cara Pakai

### Quick Start (Menu)
```bash
START_COMPETITION.bat
```

### Training Multi-Agent
```bash
# Train 3 agents, 5 rounds, 10000 steps per round
python 1_COMPETITION_MASTER.py --mode train --agents 3 --rounds 5 --steps 10000
```

### Lihat Leaderboard
```bash
python 3_COMPETITION_VIEWER.py --mode view --top 20
```

### Live Dashboard
```bash
python 3_COMPETITION_VIEWER.py --mode live --refresh 3
```

### Head-to-Head Battle
```bash
# Auto top 2 agents
python 2_BATTLE_ARENA.py --mode 1v1

# Specific agents
python 2_BATTLE_ARENA.py --mode 1v1 --agent1 Racer_v1_1 --agent2 Racer_v1_2 --steps 3000
```

### Championship Bracket
```bash
python 2_BATTLE_ARENA.py --mode bracket
```

## 🧠 Algoritma

### Reward Function
```
reward = +1.0              # Per step (tetap bergerak)
        +20.0             # New waypoint discovery
        +[0-5]            # Speed reward (context-aware)
        +[0-5]            # Following memory route
        +[0-2]            # Center-line bonus

penalty = -50/step        # Off-road (akumulatif)
         -100             # Red light violation
         -100             # Collision (per event)
```

### Decision Making Logic
```
IF danger_ahead > 0.5 OR (not on_road AND speed > 10) OR nearby_hazard:
    throttle = min(throttle, 0.2)    # Force slow down
    brake = max(brake, 0.3)           # Apply brake
    # Agent "ragu-ragu" untuk mikir
```

### Score Calculation
```
score = total_distance
      - (total_penalty × 0.1)
      - ((100 - safety_score) × 10)
```

## 📊 Metrics

| Metric | Deskripsi |
|--------|-----------|
| Distance | Total jarak tempuh (meter) |
| Penalty | Akumulasi semua penalti |
| Collisions | Jumlah tabrakan |
| Off-roads | Jumlah keluar jalur |
| Safety Score | Skor keselamatan (0-100) |
| Waypoints | Jumlah waypoint dikunjungi |

## 🏆 Tips

1. **Training pertama**: Mulai dengan 2-3 agents
2. **Steps per round**: 10000 untuk quick test, 50000+ untuk serious training
3. **Monitoring**: Pakai live dashboard untuk progress real-time
4. **Memory check**: Lihat memory agent untuk understand behavior patterns

## 🔧 Dependencies

```
pip install stable-baselines3 gymnasium carla numpy opencv-python rich torch
```
