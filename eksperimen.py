"""Pengujian awal UTS: 10 run dengan seed berbeda untuk setiap kombinasi operator.

Mencatat per run: fitness terbaik, fitness rata-rata populasi akhir, generasi valid pertama, waktu.
Hasil: output/eksperimen/hasil_run.csv, ringkasan.csv, konvergensi_rata2.png, generasi_valid.png

    python eksperimen.py            # 10 seed x 4 kombinasi
    python eksperimen.py --runs 30  # lebih banyak run
"""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import muat_data
from src.fitness import pelanggaran_lunak
from src.ga import KonfigurasiAG, jalankan_ag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--gen", type=int, default=300)
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--out", default="output/eksperimen")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = muat_data("data")
    seeds = [1000 + i for i in range(a.runs)]
    kombinasi = list(product(["turnamen", "roulette"], ["uniform", "satu_titik"]))

    baris, kurva = [], {}
    for sel, cx in kombinasi:
        label = f"{sel} + {cx}"
        kurva[label] = []
        for sd in seeds:
            cfg = KonfigurasiAG(ukuran_populasi=a.pop, maks_generasi=a.gen, seleksi=sel, crossover=cx, seed=sd)
            h = jalankan_ag(data, cfg)
            kurva[label].append(h.riwayat_terbaik)
            baris.append({
                "konfigurasi": label, "seleksi": sel, "crossover": cx, "seed": sd,
                "fitness_terbaik": h.fitness_terbaik,
                "pelanggaran_terbaik": h.pelanggaran_terbaik,
                "fitness_rata2_akhir": h.riwayat_rata2[-1],
                "generasi_valid_pertama": h.generasi_valid_pertama,
                "valid": h.generasi_valid_pertama is not None,
                **pelanggaran_lunak(h.terbaik, data),
                "waktu_detik": round(h.waktu_detik, 3),
            })
            print(f"{label:24s} seed={sd} fitness={h.fitness_terbaik:.4f} "
                  f"valid_gen={h.generasi_valid_pertama}")

    df = pd.DataFrame(baris)
    df.to_csv(out / "hasil_run.csv", index=False)

    ringkas = df.groupby("konfigurasi", sort=False).agg(
        n_run=("seed", "count"),
        persen_valid=("valid", lambda s: 100 * s.mean()),
        fitness_terbaik_rata2=("fitness_terbaik", "mean"),
        fitness_terbaik_std=("fitness_terbaik", "std"),
        fitness_rata2_akhir=("fitness_rata2_akhir", "mean"),
        gen_valid_rata2=("generasi_valid_pertama", "mean"),
        gen_valid_std=("generasi_valid_pertama", "std"),
        gen_valid_min=("generasi_valid_pertama", "min"),
        gen_valid_maks=("generasi_valid_pertama", "max"),
        waktu_rata2_detik=("waktu_detik", "mean"),
    ).round(4)
    ringkas.to_csv(out / "ringkasan.csv")
    print("\n=== RINGKASAN ===")
    print(ringkas.to_string())

    # Grafik konvergensi rata-rata fitness terbaik (60 generasi pertama agar terlihat jelas)
    plt.figure(figsize=(8, 4.5))
    for label, k in kurva.items():
        m = np.mean(k, axis=0)
        plt.plot(m[:80], label=label)
    plt.xlabel("Generasi"); plt.ylabel(f"Rata-rata fitness terbaik ({a.runs} run)")
    plt.title("Konvergensi rata-rata per kombinasi operator"); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout(); plt.savefig(out / "konvergensi_rata2.png", dpi=150); plt.close()

    # Boxplot generasi valid pertama
    plt.figure(figsize=(8, 4.5))
    grup = [df[df.konfigurasi == l]["generasi_valid_pertama"].dropna() for l in kurva]
    plt.boxplot(grup, tick_labels=list(kurva))
    plt.ylabel("Generasi solusi valid pertama"); plt.title(f"Sebaran generasi valid ({a.runs} seed)")
    plt.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.savefig(out / "generasi_valid.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
