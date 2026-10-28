"""Operator genetika untuk penjadwalan kuliah.

REPRESENTASI KROMOSOM (alasan pemilihan encoding)
-------------------------------------------------
Kromosom = array integer (n_sesi, 2). Gen ke-i = (slot_global, indeks_ruang) milik sesi kuliah ke-i
(urutan sesi = urutan baris di mata_kuliah.csv). Dosen dan kelas sudah melekat pada tiap sesi,
jadi tidak perlu dikodekan ulang. Alasan memilih encoding integer langsung:
  - Setiap gen langsung bermakna jadwal: tidak perlu proses decoding yang rumit.
  - Cocok untuk crossover satu titik/uniform dan mutasi acak (nilai gen selalu tetap dalam rentang).
  - Permutasi kurang cocok: sesi tidak harus menempati slot berbeda (ruang berbeda boleh
    memakai slot yang sama), sehingga ruang pencarian bukan permutasi murni.
  - Satu gen membawa pasangan (slot, ruang) sehingga crossover memindahkan keduanya bersamaan.
"""
import numpy as np


def inisialisasi_populasi(ukuran, data, rng):
    """Populasi acak (P, n_sesi, 2): slot di [0, n_slot), ruang di [0, n_ruang)."""
    slot = rng.integers(0, data.n_slot, size=(ukuran, data.n_sesi))
    ruang = rng.integers(0, data.n_ruang, size=(ukuran, data.n_sesi))
    return np.stack([slot, ruang], axis=2)


# ---------------------------- SELEKSI (mengembalikan indeks orang tua) ----------------------------
def seleksi_turnamen(fitness, n, rng, k=3):
    """Tiap orang tua = pemenang dari k peserta acak."""
    peserta = rng.integers(0, len(fitness), size=(n, k))
    menang = np.argmax(fitness[peserta], axis=1)
    return peserta[np.arange(n), menang]


def seleksi_roulette(fitness, n, rng, k=None):
    """Peluang terpilih sebanding dengan fitness (fitness selalu > 0)."""
    p = fitness / fitness.sum()
    return rng.choice(len(fitness), size=n, p=p)


# ---------------------------- CROSSOVER (p1, p2: (n_pasang, n_sesi, 2)) ----------------------------
def crossover_satu_titik(p1, p2, rng):
    """Titik potong acak di antara gen; sebelum titik ikut p1, sesudahnya ikut p2 (dan sebaliknya)."""
    n_pasang, n_sesi = p1.shape[0], p1.shape[1]
    titik = rng.integers(1, n_sesi, size=(n_pasang, 1))
    kiri = (np.arange(n_sesi)[None, :] < titik)[:, :, None]
    return np.where(kiri, p1, p2), np.where(kiri, p2, p1)


def crossover_uniform(p1, p2, rng):
    """Tiap gen (pasangan slot+ruang) ditukar dengan peluang 0.5."""
    tukar = (rng.random(p1.shape[:2]) < 0.5)[:, :, None]
    return np.where(tukar, p2, p1), np.where(tukar, p1, p2)


# ---------------------------- MUTASI ----------------------------
def mutasi(pop, pm, data, rng):
    """Mutasi acak per gen dengan peluang pm: ganti slot, ruang, atau keduanya (sama rata)."""
    pop = pop.copy()
    terpilih = rng.random(pop.shape[:2]) < pm
    jenis = rng.integers(0, 3, size=pop.shape[:2])          # 0 = slot, 1 = ruang, 2 = keduanya
    slot_baru = rng.integers(0, data.n_slot, size=pop.shape[:2])
    ruang_baru = rng.integers(0, data.n_ruang, size=pop.shape[:2])
    ubah_slot = terpilih & (jenis != 1)
    ubah_ruang = terpilih & (jenis != 0)
    pop[:, :, 0] = np.where(ubah_slot, slot_baru, pop[:, :, 0])
    pop[:, :, 1] = np.where(ubah_ruang, ruang_baru, pop[:, :, 1])
    return pop


SELEKSI = {"turnamen": seleksi_turnamen, "roulette": seleksi_roulette}
CROSSOVER = {"satu_titik": crossover_satu_titik, "uniform": crossover_uniform}
