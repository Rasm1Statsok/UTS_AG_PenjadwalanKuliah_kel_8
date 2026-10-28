"""Menjalankan AG satu kali dan menghasilkan jadwal terbaik.

Contoh:
    python main.py
    python main.py --seleksi roulette --crossover satu_titik --pop 50 --gen 500 --seed 7
"""
import argparse

import pandas as pd

from src.data_loader import muat_data
from src.ga import KonfigurasiAG, jalankan_ag
from src.output import ekspor, plot_konvergensi, ringkasan_pelanggaran, tabel_grid


def main():
    p = argparse.ArgumentParser(description="Penjadwalan kuliah dengan Algoritma Genetika")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="output")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gen", type=int, default=300)
    p.add_argument("--pc", type=float, default=0.8)
    p.add_argument("--pm", type=float, default=0.05)
    p.add_argument("--seleksi", choices=["turnamen", "roulette"], default="turnamen")
    p.add_argument("--crossover", choices=["uniform", "satu_titik"], default="uniform")
    p.add_argument("--elit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    data = muat_data(a.data)
    cfg = KonfigurasiAG(a.pop, a.gen, a.pc, a.pm, a.seleksi, 3, a.crossover, a.elit, False, a.seed)
    print(f"Data: {data.n_sesi} sesi, {len(data.dosen)} dosen, {data.n_ruang} ruang, "
          f"{data.n_hari} hari x {data.slot_per_hari} slot\nKonfigurasi: {cfg}\n")

    hasil = jalankan_ag(data, cfg, verbose=True)
    daftar = ekspor(hasil.terbaik, data, a.out)
    plot_konvergensi(hasil, f"{a.out}/konvergensi.png")

    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    print("\n=== JADWAL TERBAIK (hari x slot, semua kelas) ===")
    print(tabel_grid(daftar, data).to_string())
    print(f"\nFitness terbaik      : {hasil.fitness_terbaik:.4f}")
    print(f"Generasi valid pertama: {hasil.generasi_valid_pertama}")
    print(f"Pelanggaran           : {ringkasan_pelanggaran(hasil.terbaik, data)}")
    print(f"Waktu                 : {hasil.waktu_detik:.2f} detik")
    print(f"\nFile tersimpan di folder '{a.out}/': jadwal_terbaik.csv, jadwal_terbaik.xlsx, konvergensi.png")


if __name__ == "__main__":
    main()
