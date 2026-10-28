"""Loop evolusi Algoritma Genetika (generational GA dengan elitisme)."""
import time
from dataclasses import dataclass, asdict

import numpy as np

from .data_loader import DataJadwal
from .fitness import fitness, total_pelanggaran
from . import operators as op


@dataclass
class KonfigurasiAG:
    ukuran_populasi: int = 100
    maks_generasi: int = 300
    pc: float = 0.8                  # probabilitas crossover
    pm: float = 0.05                 # probabilitas mutasi per gen
    seleksi: str = "turnamen"        # "turnamen" | "roulette"
    k_turnamen: int = 3
    crossover: str = "uniform"       # "uniform" | "satu_titik"
    n_elit: int = 2
    berhenti_saat_valid: bool = False  # True -> berhenti begitu fitness = 1
    seed: int = 42


@dataclass
class HasilAG:
    terbaik: np.ndarray
    fitness_terbaik: float
    pelanggaran_terbaik: int
    generasi_valid_pertama: int | None   # None jika tidak pernah valid
    riwayat_terbaik: list
    riwayat_rata2: list
    riwayat_terburuk: list
    waktu_detik: float
    konfigurasi: dict


def jalankan_ag(data: DataJadwal, cfg: KonfigurasiAG, verbose=False) -> HasilAG:
    rng = np.random.default_rng(cfg.seed)
    pilih = op.SELEKSI[cfg.seleksi]
    silang = op.CROSSOVER[cfg.crossover]
    N = cfg.ukuran_populasi

    mulai = time.perf_counter()
    pop = op.inisialisasi_populasi(N, data.n_sesi, data.n_slot, data.n_ruang, rng)
    fit = fitness(pop, data)

    terbaik, rata2, terburuk = [], [], []
    gen_valid = None
    best_krom, best_fit = None, -1.0

    for gen in range(cfg.maks_generasi + 1):
        # --- catat statistik generasi ini
        i_best = int(np.argmax(fit))
        if fit[i_best] > best_fit:
            best_fit, best_krom = float(fit[i_best]), pop[i_best].copy()
        terbaik.append(float(fit.max()))
        rata2.append(float(fit.mean()))
        terburuk.append(float(fit.min()))
        if gen_valid is None and best_fit == 1.0:
            gen_valid = gen
        if verbose and (gen % 25 == 0 or gen == cfg.maks_generasi):
            print(f"Gen {gen:4d} | terbaik={fit.max():.4f} rata2={fit.mean():.4f} "
                  f"pelanggaran_min={int(1 / fit.max() - 1 + 0.5)}")
        if gen == cfg.maks_generasi or (cfg.berhenti_saat_valid and gen_valid is not None):
            break

        # --- bentuk generasi baru
        elit = op.ambil_elit(pop, fit, cfg.n_elit)
        n_anak = N - len(elit)
        if cfg.seleksi == "turnamen":
            idx_induk = pilih(fit, n_anak + 1, rng, k=cfg.k_turnamen)
        else:
            idx_induk = pilih(fit, n_anak + 1, rng)

        anak = []
        for j in range(0, n_anak, 2):
            p1, p2 = pop[idx_induk[j]], pop[idx_induk[j + 1]]
            if rng.random() < cfg.pc:
                a1, a2 = silang(p1, p2, rng)
            else:
                a1, a2 = p1.copy(), p2.copy()
            anak.append(op.mutasi_acak(a1, cfg.pm, data.n_slot, data.n_ruang, rng))
            anak.append(op.mutasi_acak(a2, cfg.pm, data.n_slot, data.n_ruang, rng))
        anak = np.array(anak[:n_anak])

        pop = np.concatenate([elit, anak]) if len(elit) else anak
        fit = fitness(pop, data)

    return HasilAG(
        terbaik=best_krom,
        fitness_terbaik=best_fit,
        pelanggaran_terbaik=int(total_pelanggaran(best_krom, data)[0]),
        generasi_valid_pertama=gen_valid,
        riwayat_terbaik=terbaik,
        riwayat_rata2=rata2,
        riwayat_terburuk=terburuk,
        waktu_detik=time.perf_counter() - mulai,
        konfigurasi=asdict(cfg),
    )
