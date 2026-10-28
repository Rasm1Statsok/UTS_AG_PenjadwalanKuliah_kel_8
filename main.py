"""Jalankan satu kali AG penjadwalan kuliah lalu simpan jadwal terbaik + grafik konvergensi.

Contoh:  python main.py --pop 20 --gen 300 --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import muat_data
from src.ga import KonfigurasiAG, jalankan_ag
from src.output import ekspor, plot_konvergensi, ringkasan_pelanggaran, tabel_daftar, tabel_grid


def main():
    p = argparse.ArgumentParser(description="AG Penjadwalan Kuliah")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="output")
    p.add_argument("--pop", type=int, default=100, help="ukuran populasi")
    p.add_argument("--gen", type=int, default=300, help="maksimum generasi")
    p.add_argument("--pc", type=float, default=0.8, help="peluang crossover")
    p.add_argument("--pm", type=float, default=0.02, help="peluang mutasi per gen")
    p.add_argument("--seleksi", choices=["turnamen", "roulette"], default="turnamen")
    p.add_argument("--crossover", choices=["uniform", "satu_titik"], default="uniform")
    p.add_argument("--turnamen", type=int, default=3, help="ukuran turnamen")
    p.add_argument("--elitisme", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--berhenti-valid", action="store_true", help="berhenti saat solusi valid pertama")
    a = p.parse_args()

    data = muat_data(a.data)
    cfg = KonfigurasiAG(ukuran_populasi=a.pop, maks_generasi=a.gen, pc=a.pc, pm=a.pm,
                        seleksi=a.seleksi, crossover=a.crossover, ukuran_turnamen=a.turnamen,
                        elitisme=a.elitisme, seed=a.seed, berhenti_jika_valid=a.berhenti_valid)
    hasil = jalankan_ag(data, cfg)

    print(f"Seleksi={cfg.seleksi}, crossover={cfg.crossover}, pop={cfg.ukuran_populasi}, "
          f"pc={cfg.pc}, pm={cfg.pm}, seed={cfg.seed}")
    print(f"Fitness terbaik      : {hasil.fitness_terbaik:.4f}")
    print(f"Pelanggaran keras    : {hasil.pelanggaran_terbaik}")
    print(f"Generasi valid pertama: {hasil.generasi_valid_pertama}")
    print(f"Waktu                : {hasil.waktu_detik:.2f} detik")
    print("Rincian pelanggaran  :", ringkasan_pelanggaran(hasil.terbaik, data))

    daftar = ekspor(hasil.terbaik, data, folder=a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    plot_konvergensi(hasil, Path(a.out) / "konvergensi.png")

    pd.set_option("display.width", 200, "display.max_columns", None, "display.max_colwidth", 40)
    print("\n=== JADWAL (hari x slot) ===")
    print(tabel_grid(daftar, data).to_string())
    print(f"\nFile tersimpan di folder '{a.out}/': jadwal_terbaik.csv, jadwal_terbaik.xlsx, konvergensi.png")


if __name__ == "__main__":
    main()
