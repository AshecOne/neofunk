# Dataset untuk benchmark & fine-tuning

Data yang dibutuhkan untuk benchmark/fine-tune selalu berupa **pasangan (audio, MIDI)**:
rekaman suara + not yang benar. Dataset tidak disimpan di git (terlalu besar). Download
ke folder `data/` (sudah di-`.gitignore`) memakai perintah di bawah.

## 1. HumTrans (otomatis)

| | |
|---|---|
| Isi | ±56 jam humming, 500 komposisi, 1000 segmen, 10 penyenandung, 44.1 kHz |
| Label | MIDI per rekaman + split resmi `TRAIN` / `VALID` / `TEST` |
| Ukuran | `all_wav.zip` ±14.7 GB (butuh ±32 GB kosong saat ekstrak), `all_midi.zip` ±5.5 MB |
| Lisensi | **CC-BY-NC-4.0**: boleh untuk riset/pribadi, **tidak untuk komersial** |
| Sumber | https://huggingface.co/datasets/dadinghh2/HumTrans · paper: https://arxiv.org/abs/2309.09623 |

```bash
pip install -e ".[data]"

# lihat labelnya dulu (cepat, ±6 MB)
neofunk data humtrans --dir data/humtrans --no-audio

# lengkap dengan audio (±14.7 GB download)
neofunk data humtrans --dir data/humtrans
```

Hasilnya:

```
data/humtrans/
  train_valid_test_keys.json   split resmi
  midi/...                     label .mid
  wav/...                      audio .wav (nama file sama dengan .mid-nya)
  dynamic_humtrans/            label onset/offset yang dikoreksi (Gupta et al., 2024)
```

Nama file: `personID_musicID_segmentID_repetitionID[_U/D/DD/DDD]` (sama untuk .wav dan .mid).

Catatan kualitas label: paper *Dynamic HumTrans* (https://arxiv.org/abs/2410.05455)
menemukan sebagian onset/offset HumTrans kurang pas. Karena itu label koreksinya ikut
di-clone ke `dynamic_humtrans/`. Untuk benchmark onset ketat, bandingkan dengan
`--onset-tol 0.1` juga.

Python:

```python
from neofunk.datasets import load_humtrans
test = load_humtrans("data/humtrans", "TEST")   # list[Pair(name, audio, midi, split)]
```

## 2. Rekamanmu sendiri (paling penting untuk benchmark)

Taruh di satu folder, nama audio dan MIDI harus sama:

```
my_eval/
  lagu1.wav   lagu1.mid
  solo2.wav   solo2.mid
```

Lalu `neofunk eval --data my_eval --backends pyin,crepe,rosvot`.
20–50 potongan pendek (5–20 detik) sudah cukup untuk benchmark yang berarti, dan
juga berguna untuk fine-tune.

## 3. Dataset lain (manual, untuk nanti)

| Dataset | Isi | Akses |
|---|---|---|
| M4Singer | nyanyi Mandarin + not + lirik (dipakai melatih ROSVOT) | form permintaan di https://github.com/M4Singer/M4Singer |
| MIR-ST500 | 500 lagu pop, label not vokal | label publik, audio harus diambil dari YouTube |
| MUSAN | noise/musik/speech untuk augmentasi "tahan noise" | https://www.openslr.org/17/ |

## Fine-tuning: rencana

1. Benchmark dulu (`neofunk eval`) di HumTrans TEST + rekamanmu: lihat backend mana
   yang terbaik dan di mana salahnya (onset? pitch? oktaf?).
2. Kalau ROSVOT yang terbaik tapi masih kurang untuk gayamu: fine-tune ROSVOT dari
   checkpoint resminya dengan HumTrans TRAIN (+ rekamanmu). Repo ROSVOT sudah punya
   script training (`tasks/run.py --config configs/rosvot.yaml`). Yang perlu dibuat: script
   konversi HumTrans → format manifest ROSVOT. Humming tidak punya lirik, jadi batas
   "kata" bisa disamakan dengan batas not. Model ±12 M parameter muat di 16 GB VRAM.
3. Ukur ulang dengan `neofunk eval`. Fine-tune dianggap berhasil kalau note F1 di
   rekamanmu naik, bukan cuma di HumTrans.
