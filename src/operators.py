"""Operator genetika yang ditulis sendiri (tanpa DEAP/PyGAD).

Semua fungsi menerima `rng` (np.random.Generator) agar hasil dapat direproduksi lewat seed.
"""
import numpy as np


# ---------------------------------------------------------------- inisialisasi
def inisialisasi_populasi(ukuran, n_sesi, n_slot, n_ruang, rng):
    """Populasi acak: tiap gen diberi slot dan ruang acak."""
    pop = np.empty((ukuran, n_sesi, 2), dtype=int)
    pop[:, :, 0] = rng.integers(0, n_slot, size=(ukuran, n_sesi))
    pop[:, :, 1] = rng.integers(0, n_ruang, size=(ukuran, n_sesi))
    return pop


# ---------------------------------------------------------------- seleksi
def seleksi_roulette(fit, n_pilih, rng):
    """Roulette wheel: peluang terpilih sebanding dengan fitness.

    Dibuat manual: hitung kumulatif peluang, lalu cari posisi bilangan acak di 'roda'.
    """
    peluang = fit / fit.sum()
    kumulatif = np.cumsum(peluang)
    r = rng.random(n_pilih)
    idx = np.searchsorted(kumulatif, r)
    return np.minimum(idx, len(fit) - 1)


def seleksi_turnamen(fit, n_pilih, rng, k=3):
    """Turnamen: ambil k individu acak, pemenangnya yang fitness-nya tertinggi."""
    peserta = rng.integers(0, len(fit), size=(n_pilih, k))
    pemenang = np.argmax(fit[peserta], axis=1)
    return peserta[np.arange(n_pilih), pemenang]


# ---------------------------------------------------------------- crossover
def crossover_satu_titik(p1, p2, rng):
    """Potong di satu titik acak, tukar bagian ekor. Pasangan (slot, ruang) tidak dipisah."""
    titik = rng.integers(1, len(p1))
    a1 = np.concatenate([p1[:titik], p2[titik:]])
    a2 = np.concatenate([p2[:titik], p1[titik:]])
    return a1, a2


def crossover_uniform(p1, p2, rng):
    """Tiap gen dipilih dari induk 1 atau induk 2 dengan peluang 0.5."""
    mask = rng.random(len(p1)) < 0.5
    a1 = np.where(mask[:, None], p1, p2)
    a2 = np.where(mask[:, None], p2, p1)
    return a1, a2


# ---------------------------------------------------------------- mutasi
def mutasi_acak(krom, pm, n_slot, n_ruang, rng):
    """Setiap gen bermutasi dengan peluang pm: slot dan/atau ruangnya diganti nilai acak."""
    krom = krom.copy()
    for i in np.flatnonzero(rng.random(len(krom)) < pm):
        jenis = rng.integers(0, 3)          # 0 = ganti slot, 1 = ganti ruang, 2 = ganti keduanya
        if jenis in (0, 2):
            krom[i, 0] = rng.integers(0, n_slot)
        if jenis in (1, 2):
            krom[i, 1] = rng.integers(0, n_ruang)
    return krom


# ---------------------------------------------------------------- elitisme
def ambil_elit(pop, fit, n_elit):
    """Kembalikan salinan n_elit individu terbaik."""
    if n_elit <= 0:
        return pop[:0].copy()
    idx = np.argsort(-fit, kind="stable")[:n_elit]
    return pop[idx].copy()


SELEKSI = {"roulette": seleksi_roulette, "turnamen": seleksi_turnamen}
CROSSOVER = {"satu_titik": crossover_satu_titik, "uniform": crossover_uniform}
