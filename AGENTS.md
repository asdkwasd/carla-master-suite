# CARLA Self-Driving — MASTER AGENT GUIDE (universal / provider-agnostic)
<!-- Simpan sebagai AGENTS.md + CLAUDE.md + GEMINI.md + MASTER_PROMPT.txt (isi sama). -->
<!-- Dibaca oleh AI CLI mana pun. Sumber kebenaran = file ini + STATE.md + git. -->

## 0. IDENTITAS & ENVIRONMENT (JANGAN TEBAK YANG LAIN)
- OS: Windows
- Repo (kanonik): D:\AI\traffic_vision\master_suite
- Venv: D:\AI\traffic_vision\venv   (aktivasi: source ../venv/Scripts/activate)
- CARLA: D:\AI\traffic_vision\CARLA_0.9.16   (versi 0.9.16, UE4)
- GPU: RTX 4080 Laptop, 12 GB VRAM
- CARLA RPC port: localhost:2000

## 1. RESUME RULE (LAKUKAN DULU — hemat request)
1. Baca HANYA: STATE.md + 5 baris terakhir PROGRESS.jsonl.
   JANGAN eksplor repo / baca core/ kecuali NEXT ACTION memang mengeditnya.
2. Restate 3 baris: phase | next action | expected outcome. Lalu langsung kerja.
3. Kerjakan TEPAT SATU next action (bounded), lalu:
   - update STATE.md (ringkas), append 1 baris ke PROGRESS.jsonl,
   - git add -A && git commit -m "..." && git push.
4. Tiap cycle WAJIB hasilkan 1 perubahan kode ATAU 1 hasil eval.
   DILARANG cycle yang cuma menulis ulang dokumentasi.
5. Live monitoring BUKAN tugasmu (user jalankan watch.py sendiri).
   JANGAN spawn subagent untuk memantau GPU/CARLA/log.

## 2. TUGAS (dua lapis, berurutan)
(A) STABILKAN codebase yang ADA supaya mobil menyetir andal & training tidak
    pernah freeze  (Phase -1).
(B) BARU evolusi jadi stack yang kompetensi menyetirnya MENYAMAI/MELAMPAUI
    Waymo/Tesla, dibuktikan benchmark closed-loop yang reproducible.

## 3. KEEP (bagian bagus stack yang ada — jangan buang tanpa alasan)
PPO (stable_baselines3) + GrandmasterFeatureExtractor (semseg cam 80x60 CNN +
16-sector LiDAR + ~28-dim state fusion); SafetyShield, CurriculumManager(4),
RecallMemory, RecoveryModule, ADR, HER, WorldModel, EpisodicMemory, self-play
Rival, ONNX->TensorRT deploy, Observer+Monitor.
Entrypoint kanonik: 1_TRAIN_MASTER_v15.py, 3_TEST_LIVE_v15.py,
4_DEPLOY_MASTER_v15.py, dan core/ (constants.py, cost_function.py, safety.py).

## 4. PRINSIP NON-NEGOTIABLE
1. RESEARCH-FIRST: sebelum koding komponen, baca 1 paper peer-review
   (Scopus/IEEE/Elsevier/Springer) + arXiv + repo resminya; tulis ringkasan 5 baris
   (metode, skor CARLA, compute, kenapa). Perluas SOTA, jangan menebak.
2. CLOSED-LOOP = KEBENARAN: north-star = Driving Score = Route Completion x
   Infraction multiplier di CARLA Leaderboard 2.0/2.1 + Bench2Drive.
   Open-loop cuma diagnostik.
3. STAND ON GIANTS THEN BEAT THEM: reproduksi baseline kuat dulu, catat skornya,
   baru improve. Baseline: TransFuser++ (carla_garage), SimLingo, Raw2Drive,
   Think2Drive, CaRL, PDM-lite (teacher), Roach, LMDrive/LangAuto.
4. REPRODUCIBILITY: pin versi, seed semua, log config + git SHA, simpan eval JSON
   + video rute per run.
5. SAFETY-AS-METRIC: tiap collision/red-light/off-road/blocked = regresi kelas satu;
   jangan eksploitasi early-stop scoring; laporkan jujur.

## 5. TARGET ARSITEKTUR (bangun bertahap, ablate tiap blok)
- SENSORS: multi-camera (vision-centric); LiDAR/radar opsional via flag config.
- WORLD MODEL: BEV encoder + 3D Occupancy net + head object/map/traffic-light.
- SYSTEM 1 (fast): policy reaktif latensi rendah -> kontrol >=10 Hz; imitation dari
  PDM-lite/Roach + RL fine-tune (Raw2Drive/Think2Drive/CaRL).
- SYSTEM 2 (slow): reasoning VLA/LLM (SimLingo/LMDrive) untuk keputusan
  horizon-panjang + penjelasan "kenapa".
- SAFETY SHIELD: checker terverifikasi (collision/TTC, aturan, comfort) yang bisa veto.

## 6. WORKFLOW
Phase -1 Stabilkan: tanpa freeze; mobil jalan > 200 m di Stage 1 lintas 3 seed.
Phase 0  Smoke-test env (CARLA 0.9.16 + Leaderboard 2.0/2.1 + Bench2Drive).
Phase 1  Data engine: expert PDM-lite/Roach, tambang failure long-tail.
Phase 2  Reproduksi TransFuser++; catat DS/RC/infraction per rute.
Phase 3  Perbaiki persepsi (occupancy+BEV); ablation vs baseline.
Phase 4  RL fine-tune System-1 + tambah System-2 VLA + safety shield.
Phase 5  Eval closed-loop (LB2/2.1 + Bench2Drive + LangAuto); taksonomi error.
Phase 6  Perbaiki klaster skenario terburuk; ulang Phase 5.
Phase 7  Robustness (town baru, malam/hujan, sensor noise) + rencana sim-to-real.

## 7. ATURAN GPU (RTX 4080 Laptop 12 GB — time-slicing)
- CARLA hanya hidup saat (a) collect data dan (b) closed-loop eval; saat itu training di-pause.
- KILL CARLA sebelum training apa pun supaya 100% GPU buat epoch.
- Pakai AMP/bf16, gradient checkpointing + accumulation, torch.compile,
  checkpoint tiap N epoch. JANGAN train & eval bersamaan.
- Start CARLA (Windows):
    cd D:\AI\traffic_vision\CARLA_0.9.16
    start "" "CarlaUE4.exe" -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=0
- Kill CARLA (Windows):
    taskkill /IM CarlaUE4.exe /F

## 8. DEFINITION OF DONE (jangan klaim lebih awal)
Phase -1 lulus; baseline direproduksi DAN dikalahkan di rute SAMA; mean DS di
LB2/2.1 held-out >= target DAN > baseline lintas >=3 seed dengan variance;
0 collision tak terjelaskan; >=1 run generalisasi town-baru; bundle reproducibility
lengkap (config, weights, eval JSON, video, RESULTS.md, REFERENCES.md); baton
(STATE.md/HANDOFF.md/PROGRESS.jsonl/NEXT_ACTIONS.md) terkini & sudah push.

## 9. GUARDRAILS
Jangan mengarang skor. Commit kecil reversible + CHANGELOG.md. Jangan hapus file
sampai nilai uniknya dipindah & di-commit. Kalau metode SOTA malah jelek di sini,
laporkan + pivot (dokumentasikan negative result). Jangan ubah STATE_DIM atau
palette semantic-camera tanpa update core/constants.py + migrasi checkpoint.
Selalu tinggalkan repo resumable (STATE.md + HANDOFF.md ter-commit) sebelum berhenti
karena alasan APA PUN.

## 10. BASELINE & REFERENSI (baca sebelum koding; simpan ke REFERENCES.md)
- CARLA Leaderboard 2.0/2.1 — metrik & rute resmi.
- carla_garage / TransFuser++ — starter kit + SOTA LB2.
- SimLingo — VLA camera-only, juara CARLA Challenge 2024 (arXiv:2503.09594).
- Raw2Drive — RL + aligned world models (NeurIPS 2025).
- Think2Drive — latent world-model RL untuk CARLA v2.
- CaRL — scalable planning policies (arXiv:2504.17838).
- PDM-lite — rule-based planner (teacher/expert).
- Roach — RL coach untuk imitation (ICCV 2021, arXiv:2108.08265).
- LMDrive — closed-loop LLM driving + LangAuto (CVPR 2024, arXiv:2312.07488).
- Bench2Drive — benchmark closed-loop end-to-end.
- Waymo Foundation Model — arsitektur Think Fast/Think Slow (System 1 & 2).

## 11. TELEMETRI (/telemetry/schema.json)
```json
{
  "run_id": "string",
  "git_sha": "string",
  "timestamp": "ISO-8601",
  "benchmark": "LB2.1 | Bench2Drive | LangAuto",
  "seed": 0,
  "aggregate": { "driving_score": 0.0, "route_completion": 0.0, "infraction_penalty": 0.0 },
  "infractions_per_km": {
    "collisions_pedestrian": 0.0, "collisions_vehicle": 0.0, "collisions_layout": 0.0,
    "red_light": 0.0, "stop_sign": 0.0, "off_road": 0.0, "agent_blocked": 0.0
  },
  "top_failures": [ {"scenario": "string", "route": "string", "hypothesis": "string"} ],
  "system2_narration": "penjelasan bahasa alami manuver berisiko terakhir",
  "delta_since_last": "apa yang berubah (config/diff)",
  "next_action": "string",
  "source_backing_next_action": "URL paper/forum"
}
```

## 12. UNIVERSAL / PROVIDER-AGNOSTIC (anti bergantung 1 sistem)
Otak agent = file ini + STATE.md + PROGRESS.jsonl + git (teks biasa).
Provider (Claude/Codex/Gemini/aider) cuma "tangan". Ganti kapan saja tanpa
kehilangan progres, selama tiap cycle progres selalu turun ke STATE.md + git.

Adapter tiap provider (headless / non-interaktif):
- Claude Code : claude -p "$P" --dangerously-skip-permissions --output-format stream-json
- OpenAI Codex: codex exec --full-auto "$P"
- Gemini CLI  : gemini -y -p "$P"
- aider       : aider --yes --message "$P"
- Notion AI   : BUKAN CLI coding; untuk baca STATE.md/planning/review, bukan loop.
(Flag bisa beda antar versi — cek --help sekali lalu kunci di adapter.)

## 13. STATE.md (template baton tunggal — jaga < 40 baris)
```markdown
# STATE.md  (baton tunggal)
REPO: D:/AI/traffic_vision/master_suite
UPDATED: <tanggal>
PHASE: -1 (Stabilize) — BELUM lulus

## NEXT ACTION (kerjakan TEPAT SATU)
> <deskripsi next action>
> File: <file yang perlu diedit>

## RESUME COMMAND (persis)
cd D:/AI/traffic_vision/master_suite && source ../venv/Scripts/activate && python 1_TRAIN_MASTER_v15.py

## METRIK TERAKHIR (fakta, jangan dikarang)
steps=... | reward=... | dist=... | GPU=... | stage=...

## GATE LULUS PHASE -1
3 seed, mean dist > 200m di Stage 1, tanpa freeze.
```
