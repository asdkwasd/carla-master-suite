# PROJECT RECALL: CARLA GRANDMASTER AI v8.0

## Arsitektur Terakhir (Phase: Evolution)
- **Lokasi:** `D:\AI\traffic_vision\master_suite`
- **Mode:** Symmetric Self-Play (Ego AI vs Rival AI menggunakan otak yang sama).
- **Visi:** Semantic Segmentation (Aspal=Hijau, Lainnya=Geometri Merah/Gelap).
- **Reward:** 
  - `copy_reward`: ATM (Amati, Tiru, Modifikasi) mengikuti steering ideal.
  - `speed_reward`: Bonus ngebut di aspal & persimpangan.
  - `ghost_reward`: Bonus mengikuti rekor jalur terbaik masa lalu.
- **State Dim:** 11 (Termasuk radar aspal ke depan).

## Cara Melanjutkan (Resume)
1. Cek folder `checkpoints/` untuk model `.zip` terbaru.
2. Jalankan `1_TRAIN_MASTER.py` untuk melanjutkan latihan.
3. Jalankan `2_MONITOR_GRAPH.py` untuk melihat grafik.
4. Jalankan `3_TEST_LIVE.py` untuk melihat balapan dari atas (Drone View).

## Pesan Penting
Jangan mengubah `STATE_DIM` atau `palette` warna kamera tanpa instruksi user, karena akan merusak kompatibilitas model yang sedang dilatih.
