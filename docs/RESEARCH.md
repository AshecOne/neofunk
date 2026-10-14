# Riset: suara (humming/nyanyi) → not

Masalah ini di dunia riset disebut **Singing Voice Transcription (SVT)** atau
**Humming/Query-by-Humming transcription**, bagian dari *Automatic Music Transcription*
(AMT). Karena suara mulut itu **monofonik** (satu nada dalam satu waktu), masalahnya jauh
lebih mudah daripada transkripsi piano atau band lengkap. Itu sebabnya sistem seperti ini
bisa jalan di laptop atau GPU kelas menengah.

## Pipeline umum

1. **Pitch tracking (f0 estimation)**: frekuensi dasar per frame (~10 ms).
   - Klasik: YIN, **pYIN** (probabilistik + HMM, ada di librosa).
   - Neural: **CREPE**, **RMVPE** (tahan noise, populer di komunitas RVC/SVC),
     **FCPE** (cepat).
2. **Segmentasi not**: kapan not mulai/berhenti, dan pitch bulatnya berapa. Bagian ini
   yang paling susah karena ada vibrato, *scoop* (naik ke nada), glide, konsonan,
   dan nada yang diulang tanpa jeda.
   - Heuristik (yang dipakai neofunk sekarang): median filter, ambang perubahan pitch,
     onset detection, durasi minimum.
   - Model note-level: memprediksi onset/offset/pitch langsung (Basic Pitch, SOME, ROSVOT).
3. **Musikalisasi**: koreksi tuning global, deteksi key (Krumhansl-Schmuckler), estimasi
   tempo, kuantisasi ke grid ketukan.
4. **Ekspor**: MIDI (DAW) dan MusicXML (partitur).

## Perbandingan model/tool yang relevan

| Model / tool | Jenis | Kelebihan | Catatan |
|---|---|---|---|
| **pYIN** (librosa) | pitch tracker klasik | tanpa model, CPU, sudah di neofunk | sedikit kalah di rekaman yang bising |
| **CREPE / torchcrepe** | pitch tracker CNN | akurat & robust, sudah di neofunk (`--backend crepe`) | model "full" ±85 MB, ringan untuk RTX 4060 |
| **RMVPE** | pitch tracker | sangat tahan noise/napas, dipakai SOME & RVC | kandidat backend berikutnya |
| **TorchFCPE** | pitch tracker | sangat cepat, ada output MIDI | MIDI-nya hasil kuantisasi non-neural dari f0 |
| **Spotify Basic Pitch** | audio→MIDI (polifonik) | gampang dipakai, ada pitch bend, sudah di neofunk (`--backend basic-pitch`) | untuk humming kadang "choppy" |
| **SOME** (openvpi) | MIDI extractor khusus nyanyi | cepat (~300x real-time di 3080Ti), dibuat untuk label DiffSinger | status beta |
| **ROSVOT** (ACL 2024) | SVT note-level | SOTA, robust untuk input bersih maupun bising | lebih berat, kodenya untuk riset |
| **VocalParse** (2026, preprint) | SVT berbasis Large Audio LLM | menyatukan lirik + not | masih baru, jauh lebih berat |

**Kesimpulan:** untuk kebutuhanmu (melodi mulut → MIDI → bikin lagu) **tidak perlu
melatih model dari nol**. Pitch tracker pretrained ditambah segmentasi yang bagus sudah
memberi hasil yang bisa dipakai. Training/fine-tuning baru berguna nanti, untuk menyesuaikan
ke gaya suaramu (misalnya "solo gitar" mulut yang cepat dengan banyak glide).

## Dataset (untuk evaluasi & fine-tuning)

- **HumTrans**: ±56 jam humming, 500 komposisi, label MIDI, ada split train/valid/test.
- **Dynamic HumTrans**: perbaikan label onset/offset HumTrans + baseline CNN.
- **MIR-ST500 / M4Singer / Opencpop**: dataset nyanyi dengan label not (pernah dipakai
  oleh ROSVOT/SOME).
- **Rekamanmu sendiri**: 20–50 potongan yang kamu koreksi manual di DAW sudah sangat
  berharga untuk evaluasi, dan juga untuk fine-tuning.

Metrik standar: `mir_eval.transcription` (precision/recall/F1 untuk onset ±50 ms, pitch
±50 cents, dan offset).

## Rencana teknis berikutnya

1. Tambah backend **RMVPE** (pitch) dan **SOME** (note-level) sebagai opsi `--backend`.
2. Script evaluasi `neofunk eval` memakai HumTrans subset + `mir_eval`, supaya tiap
   perubahan heuristik/model terukur.
3. Fine-tune model note-level kecil (CNN/Conformer, < 20 M parameter) di HumTrans dan
   rekamanmu. Training seperti ini muat di RTX 4060 (pakai mixed precision, batch kecil).
4. Tahap "jadi lagu": dari MIDI melodi → deteksi progresi chord yang cocok, bassline,
   pola drum (rule-based atau model seperti music-generation transformer), ekspor
   multi-track MIDI.

## Referensi

- Spotify Basic Pitch: https://github.com/spotify/basic-pitch · ulasan: https://gigazine.net/gsc_news/en/20221004-basic-pitch
- SOME (Singing-Oriented MIDI Extractor): https://github.com/openvpi/SOME
- ROSVOT, *Robust Singing Voice Transcription Serves Synthesis* (ACL 2024): https://arxiv.org/abs/2405.09940 · https://aclanthology.org/2024.acl-long.526
- HumTrans dataset: https://arxiv.org/abs/2309.09623 · https://huggingface.co/datasets/dadinghh2/HumTrans
- Dynamic HumTrans: https://arxiv.org/abs/2410.05455
- FCPE: https://huggingface.co/niobures/FCPE
- CREPE: Kim et al., *CREPE: A Convolutional Representation for Pitch Estimation* (ICASSP 2018), torchcrepe: https://github.com/maxrmorrison/torchcrepe
- pYIN: Mauch & Dixon, *pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions* (ICASSP 2014)
- Krumhansl-Schmuckler key finding: Krumhansl, *Cognitive Foundations of Musical Pitch* (1990)
