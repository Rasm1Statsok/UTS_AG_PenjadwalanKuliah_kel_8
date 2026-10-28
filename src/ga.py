"""Algoritma Genetika standar untuk penjadwalan kuliah (tahap UTS)."""
import time
from dataclasses import dataclass, field

import numpy as np

from .data_loader import DataJadwal
from .fitness import evaluasi_populasi
from .operators import CROSSOVER, SELEKSI, inisialisasi_populasi, mutasi


@dataclass
class KonfigurasiAG:
    ukuran_populasi: int = 100
    maks_generasi: int = 300
    pc: float = 0.8                  # peluang crossover per pasangan orang tua
    pm: float = 0.02                 # peluang mutasi per gen
    seleksi: str = "turnamen"        # "turnamen" | "roulette"
    crossover: str = "uniform"       # "uniform" | "satu_titik"
    ukuran_turnamen: int = 3
    elitisme: int = 2                # jumlah individu terbaik yang disalin langsung
    seed: int = None
    berhenti_jika_valid: bool = False  # False -> selalu jalan sampai maks_generasi (grafik sama panjang)


@dataclass
class HasilAG:
    terbaik: np.ndarray              # kromosom terbaik sepanjang run, shape (n_sesi, 2)
    fitness_terbaik: float
    pelanggaran_terbaik: int         # total pelanggaran keras pada kromosom terbaik
    riwayat_terbaik: list = field(default_factory=list)
    riwayat_rata2: list = field(default_factory=list)
    riwayat_terburuk: list = field(default_factory=list)
    generasi_valid_pertama: int = None   # None jika tidak pernah valid (generasi 0 = populasi awal)
    waktu_detik: float = 0.0
    konfigurasi: KonfigurasiAG = None


def jalankan_ag(data: DataJadwal, cfg: KonfigurasiAG = None) -> HasilAG:
    cfg = cfg or KonfigurasiAG()
    if cfg.seleksi not in SELEKSI:
        raise ValueError(f"seleksi harus salah satu dari {list(SELEKSI)}")
    if cfg.crossover not in CROSSOVER:
        raise ValueError(f"crossover harus salah satu dari {list(CROSSOVER)}")
    if not 0 <= cfg.elitisme < cfg.ukuran_populasi:
        raise ValueError("elitisme harus di antara 0 dan ukuran_populasi-1")

    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    pilih, silang = SELEKSI[cfg.seleksi], CROSSOVER[cfg.crossover]
    P = cfg.ukuran_populasi

    pop = inisialisasi_populasi(P, data, rng)
    pelanggaran, fit = evaluasi_populasi(pop, data)

    h = HasilAG(terbaik=pop[np.argmax(fit)].copy(), fitness_terbaik=float(fit.max()),
                pelanggaran_terbaik=int(pelanggaran[np.argmax(fit)]), konfigurasi=cfg)

    def catat(gen):
        i = int(np.argmax(fit))
        h.riwayat_terbaik.append(float(fit[i]))
        h.riwayat_rata2.append(float(fit.mean()))
        h.riwayat_terburuk.append(float(fit.min()))
        if fit[i] > h.fitness_terbaik:
            h.terbaik, h.fitness_terbaik, h.pelanggaran_terbaik = pop[i].copy(), float(fit[i]), int(pelanggaran[i])
        if h.generasi_valid_pertama is None and pelanggaran[i] == 0:
            h.generasi_valid_pertama = gen

    catat(0)
    for gen in range(1, cfg.maks_generasi + 1):
        if cfg.berhenti_jika_valid and h.generasi_valid_pertama is not None:
            break
        n_anak = P - cfg.elitisme
        n_pasang = (n_anak + 1) // 2

        ortu = pilih(fit, 2 * n_pasang, rng, cfg.ukuran_turnamen)
        p1, p2 = pop[ortu[:n_pasang]], pop[ortu[n_pasang:]]
        c1, c2 = silang(p1, p2, rng)
        pakai = (rng.random(n_pasang) < cfg.pc)[:, None, None]     # tidak crossover -> anak = orang tua
        c1, c2 = np.where(pakai, c1, p1), np.where(pakai, c2, p2)

        anak = mutasi(np.concatenate([c1, c2])[:n_anak], cfg.pm, data, rng)
        elit = pop[np.argsort(-fit)[:cfg.elitisme]]
        pop = np.concatenate([elit, anak]) if cfg.elitisme else anak
        pelanggaran, fit = evaluasi_populasi(pop, data)
        catat(gen)

    h.waktu_detik = time.perf_counter() - t0
    return h
