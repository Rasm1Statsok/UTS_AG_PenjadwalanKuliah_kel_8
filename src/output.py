"""Menampilkan dan mengekspor jadwal serta grafik konvergensi."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import DataJadwal, HARI
from .fitness import pelanggaran_keras, pelanggaran_lunak


def tabel_daftar(krom, data: DataJadwal) -> pd.DataFrame:
    """Jadwal dalam bentuk daftar: satu baris per sesi kuliah."""
    baris = []
    for i, (s, r) in enumerate(krom):
        h, sl = data.indeks_ke_slot(s)
        mk = data.mata_kuliah.iloc[i]
        baris.append({
            "hari": HARI[h], "slot": sl + 1, "jam": data.jam_slot[sl],
            "kode_mk": mk["kode"], "mata_kuliah": mk["nama"], "kelas": mk["kelas"],
            "dosen": data.dosen["nama"].iloc[data.dosen_idx[i]],
            "ruang": data.ruang["kode"].iloc[r],
            "jumlah_mhs": int(data.jumlah_mhs[i]), "kapasitas_ruang": int(data.kapasitas[r]),
            "_urut": s,
        })
    return pd.DataFrame(baris).sort_values(["_urut", "ruang"]).drop(columns="_urut").reset_index(drop=True)


def tabel_grid(daftar: pd.DataFrame, data: DataJadwal, kelas=None) -> pd.DataFrame:
    """Tabel hari x slot. Isi sel: 'kode_mk kelas (ruang)'. Bisa disaring per kelas."""
    df = daftar if kelas is None else daftar[daftar["kelas"] == kelas]
    idx = [f"Slot {i + 1} ({j})" for i, j in enumerate(data.jam_slot)]
    grid = pd.DataFrame("", index=idx, columns=HARI)
    for _, b in df.iterrows():
        teks = f"{b.kode_mk} {b.mata_kuliah}" if kelas else f"{b.kode_mk} {b.kelas}"
        teks += f" ({b.ruang})"
        sel = grid.at[idx[b.slot - 1], b.hari]
        grid.at[idx[b.slot - 1], b.hari] = teks if not sel else sel + " | " + teks
    return grid


def ringkasan_pelanggaran(krom, data) -> dict:
    keras = {k: int(v[0]) for k, v in pelanggaran_keras(krom, data).items()}
    return {**keras, "total_keras": sum(keras.values()), **pelanggaran_lunak(krom, data)}


def ekspor(krom, data: DataJadwal, folder="output", nama="jadwal_terbaik"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    daftar = tabel_daftar(krom, data)
    daftar.to_csv(folder / f"{nama}.csv", index=False)
    with pd.ExcelWriter(folder / f"{nama}.xlsx") as xw:
        daftar.to_excel(xw, sheet_name="Daftar", index=False)
        tabel_grid(daftar, data).to_excel(xw, sheet_name="Grid Semua")
        for k in data.kelas["kode"]:
            tabel_grid(daftar, data, k).to_excel(xw, sheet_name=f"Kelas {k}")
        pd.DataFrame([ringkasan_pelanggaran(krom, data)]).T.rename(columns={0: "jumlah"}) \
            .to_excel(xw, sheet_name="Pelanggaran")
    return daftar


def plot_konvergensi(hasil, path, judul="Konvergensi Fitness"):
    gen = range(len(hasil.riwayat_terbaik))
    plt.figure(figsize=(8, 4.5))
    plt.plot(gen, hasil.riwayat_terbaik, label="Terbaik", lw=2)
    plt.plot(gen, hasil.riwayat_rata2, label="Rata-rata", lw=1.5)
    plt.plot(gen, hasil.riwayat_terburuk, label="Terburuk", lw=1, alpha=.6)
    if hasil.generasi_valid_pertama is not None:
        plt.axvline(hasil.generasi_valid_pertama, ls="--", c="gray",
                    label=f"Valid pertama (gen {hasil.generasi_valid_pertama})")
    plt.xlabel("Generasi"); plt.ylabel("Fitness = 1/(1+pelanggaran)")
    plt.ylim(0, 1.18); plt.title(judul); plt.grid(alpha=.3)
    plt.legend(loc="upper center", ncol=4, fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
