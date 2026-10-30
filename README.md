# Optimasi Penjadwalan Kuliah dengan Algoritma Genetika (UTS)

Proyek Kelompok Algoritma Genetika, Tahap UTS. Program ini menyusun jadwal kuliah otomatis (hari × slot × ruang) memakai Algoritma Genetika (AG) yang seluruh operator geneticanya ditulis sendiri tanpa library AG siap pakai (DEAP, PyGAD, dan sejenisnya). Library pendukung yang dipakai: NumPy, Pandas, Matplotlib, dan openpyxl.

**Program Studi Teknik Informatika, Universitas Palangkaraya, Kelompok 8**

## Anggota dan peran

| Nama | NIM | Peran |
|---|---|---|
| Muhammad Rasmi Baihaqi | 2430205030008 | Ketua / Analis Masalah |
| Erwin Bernianto | 2430105030027 | Programmer Inti AG |
| Moris Nedik | 2430105030022 | Programmer Fitness & Validasi |
| Diki Adriyan | 2430205030007 | Analis Eksperimen |
| Ahmad Faris Al Porisi | 2430205030009 | Penulis & Presenter |

## Struktur repository

```
UTS_AG_PenjadwalanKuliah_kel_8/
├── data/
│   ├── dosen.csv          8 dosen dan slot yang tidak disukai
│   ├── kelas.csv          4 kelas (TI-1, TI-3, TI-5, TI-7) dan jumlah mahasiswa
│   ├── mata_kuliah.csv    20 mata kuliah: kode, nama, sks, dosen, kelas
│   ├── ruang.csv          5 ruang dan kapasitasnya
│   └── slot.csv           5 slot waktu per hari
├── output/
│   └── eksperimen/        hasil eksperimen dan jadwal terbaik
│       ├── generasi_valid.png
│       ├── hasil_run.csv
│       ├── jadwal_terbaik.csv
│       ├── konvergensi_rata2.png
│       └── ringkasan.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py     membaca CSV dan mengubahnya menjadi indeks integer
│   ├── fitness.py         fungsi fitness, kendala keras, kendala lunak, validasi kromosom
│   ├── ga.py              loop evolusi AG (generational dengan elitisme)
│   ├── operators.py       inisialisasi, seleksi, crossover, mutasi, elitisme
│   └── output.py          tabel jadwal, ekspor CSV/Excel, grafik konvergensi
├── tests/
│   ├── conftest.py
│   └── test_fitness.py    unit test fungsi fitness dan pengecekan kendala
├── .gitignore
├── eksperimen.py          menjalankan AG berulang (4 kombinasi operator × 10 seed)
├── main.py                menjalankan AG satu kali dan menghasilkan jadwal terbaik
├── README.md
└── requirements.txt
```

## Persyaratan dan instalasi

Butuh Python 3.10 atau lebih baru dan pip. Langkah di terminal (di VS Code: Terminal, New Terminal):

```
git clone https://github.com/Rasm1Statsok/UTS_AG_PenjadwalanKuliah_kel_8.git
cd UTS_AG_PenjadwalanKuliah_kel_8
pip install -r requirements.txt
```

Jika `pip` tidak dikenali, pakai `python -m pip install -r requirements.txt` (di Windows bisa juga `py -m pip install -r requirements.txt`).

Pustaka yang dipakai: numpy, pandas, matplotlib, openpyxl.

## Data

Data adalah jadwal prodi yang disederhanakan, dengan skala UTS: 20 mata kuliah, 8 dosen, 5 ruang, 4 kelas, dan 5 hari × 5 slot waktu (25 slot per minggu).

| Berkas | Isi |
|---|---|
| `mata_kuliah.csv` | kode, nama, sks, kode dosen, kode kelas |
| `dosen.csv` | kode, nama (inisial), `slot_tidak_disukai` (format `Hari-slot`, dipisah `;`) |
| `ruang.csv` | kode ruang, kapasitas |
| `kelas.csv` | kode kelas, jumlah mahasiswa |
| `slot.csv` | nomor slot dan jam (07.00-08.40, 08.40-10.20, 10.20-12.00, 12.50-14.30, 14.30-16.10) |

Asumsi pemodelan: setiap mata kuliah dijadwalkan sebagai **satu sesi** yang menempati **satu slot**. Kolom `sks` ada di data tetapi belum dipakai pada tahap UTS.

## Cara menjalankan

Semua perintah dijalankan dari folder utama repository.

**Menjalankan AG satu kali (menghasilkan jadwal terbaik):**

```
python main.py
```

Pilihan parameter:

| Opsi | Nilai bawaan | Keterangan |
|---|---|---|
| `--data` | `data` | folder data CSV |
| `--out` | `output` | folder keluaran |
| `--pop` | 100 | ukuran populasi |
| `--gen` | 300 | jumlah generasi maksimum |
| `--pc` | 0.8 | probabilitas crossover |
| `--pm` | 0.05 | probabilitas mutasi per gen |
| `--seleksi` | `turnamen` | `turnamen` atau `roulette` |
| `--crossover` | `uniform` | `uniform` atau `satu_titik` |
| `--elit` | 2 | jumlah individu elit |
| `--seed` | 42 | seed acak (hasil dapat direproduksi) |

Contoh:

```
python main.py --seleksi roulette --crossover satu_titik --pop 50 --gen 500 --seed 7
```

**Menjalankan eksperimen (4 kombinasi operator × 10 seed):**

```
python eksperimen.py
python eksperimen.py --runs 10 --gen 300 --pop 20
```

**Menjalankan unit test:**

```
pip install pytest
python -m pytest
```

(`pytest` belum tercantum di `requirements.txt`.)

## Keluaran

| File | Dibuat oleh | Isi |
|---|---|---|
| `output/jadwal_terbaik.csv` | `main.py` | jadwal terbaik, satu baris per sesi kuliah |
| `output/jadwal_terbaik.xlsx` | `main.py` | sheet Daftar, Grid Semua, Grid per kelas, dan Pelanggaran |
| `output/konvergensi.png` | `main.py` | grafik fitness terbaik, rata-rata, dan terburuk per generasi |
| `output/eksperimen/hasil_run.csv` | `eksperimen.py` | hasil tiap run (40 baris) |
| `output/eksperimen/ringkasan.csv` | `eksperimen.py` | ringkasan per konfigurasi |
| `output/eksperimen/konvergensi_rata2.png` | `eksperimen.py` | konvergensi rata-rata tiap kombinasi operator |
| `output/eksperimen/generasi_valid.png` | `eksperimen.py` | sebaran generasi solusi valid pertama |

Jadwal terbaik juga dicetak di terminal dalam bentuk tabel hari × slot.

## Rancangan AG

**Kromosom.** Array bilangan bulat berukuran (20, 2). Gen ke-*i* adalah pasangan (slot global, indeks ruang) untuk sesi kuliah ke-*i*, dengan slot global 0 sampai 24 (hari × 5 + slot dalam hari) dan indeks ruang 0 sampai 4. Dosen dan kelas melekat pada mata kuliah sehingga tidak perlu dikodekan di kromosom.

**Kendala keras** (menentukan fitness). Pelanggaran dihitung sebagai jumlah "kelebihan" pada tiap kelompok yang bentrok (jika *k* sesi berbagi dosen/ruang/kelas dan slot yang sama, pelanggarannya *k* − 1):

1. dosen mengajar di dua tempat pada slot yang sama
2. ruang dipakai dua kelas pada slot yang sama
3. kelas mahasiswa memiliki dua sesi pada slot yang sama
4. jumlah mahasiswa melebihi kapasitas ruang

**Fungsi fitness.**

```
fitness = 1 / (1 + total pelanggaran keras)
```

Nilai 1,0 berarti jadwal valid (tanpa pelanggaran keras).

**Kendala lunak** (pada UTS hanya dihitung dan dilaporkan, belum masuk fitness): preferensi waktu dosen (sesi yang jatuh pada slot tidak disukai) dan jeda panjang (dua slot kosong berturut-turut atau lebih di antara dua sesi kelas yang sama pada hari yang sama).

**Operator genetika** (semuanya ditulis sendiri di `src/operators.py`):

| Tahap | Pilihan |
|---|---|
| Inisialisasi | populasi acak (slot dan ruang acak untuk tiap gen) |
| Seleksi | roulette wheel (buatan sendiri) dan turnamen (k = 3) |
| Crossover | satu titik dan uniform (pasangan slot-ruang tidak dipisah) |
| Mutasi | acak per gen dengan peluang `pm`: ganti slot, ganti ruang, atau keduanya |
| Elitisme | 2 individu terbaik disalin langsung ke generasi berikutnya |

## Hasil ringkas

Satu kali run dengan pengaturan bawaan (populasi 100, 300 generasi, pc 0,8, pm 0,05, turnamen + uniform, elit 2, seed 42):

| Ukuran | Nilai |
|---|---|
| Fitness terbaik | 1,0000 (jadwal valid) |
| Generasi valid pertama | 0 |
| Pelanggaran keras | 0 (dosen, ruang, kelas, kapasitas semuanya 0) |
| Preferensi dosen dilanggar | 0 |
| Jeda panjang | 7 |

Eksperimen 10 seed (1000–1009) pada setiap kombinasi operator, populasi 100 dan 300 generasi:

| Konfigurasi | Run valid | Generasi valid pertama (rata-rata ± std) | Rentang | Fitness rata-rata populasi akhir | Waktu rata-rata (detik) |
|---|---|---|---|---|---|
| turnamen + uniform | 10/10 | 2,2 ± 1,55 | 0–4 | 0,8167 | 0,114 |
| turnamen + satu titik | 10/10 | 2,1 ± 1,66 | 0–5 | 0,8429 | 0,105 |
| roulette + uniform | 10/10 | 2,2 ± 1,93 | 0–6 | 0,7698 | 0,108 |
| roulette + satu titik | 10/10 | 2,3 ± 1,89 | 0–6 | 0,7724 | 0,111 |

Catatan: seluruh 40 run menemukan jadwal valid dalam beberapa generasi pertama, dan selisih antar-konfigurasi lebih kecil daripada simpangan bakunya, sehingga pada data skala UTS ini belum ada bukti satu kombinasi operator lebih unggul dari yang lain. Perbandingan yang lebih berarti direncanakan pada tahap UAS dengan data yang lebih besar dan kendala lunak ikut dioptimalkan.

## Batasan tahap UTS

- Setiap mata kuliah dimodelkan sebagai satu sesi satu slot (`sks` belum dipakai).
- Kendala lunak belum masuk fungsi fitness; jadwal terbaik adalah individu valid pertama yang ditemukan, sehingga angka kendala lunaknya tidak dioptimalkan.
- Unit test hanya mencakup fungsi fitness dan pengecekan kendala.

## Integritas akademik

Seluruh operator AG (inisialisasi, seleksi roulette dan turnamen, crossover satu titik dan uniform, mutasi, elitisme) ditulis sendiri oleh kelompok tanpa DEAP, PyGAD, atau library AG sejenis. Penggunaan AI (Claude) untuk membantu menyusun draf README dan laporan serta memahami konsep dicantumkan pada lampiran laporan.
