"""Membaca data CSV dan mengubahnya menjadi indeks integer agar mudah diproses AG."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]


@dataclass
class DataJadwal:
    mata_kuliah: pd.DataFrame
    dosen: pd.DataFrame
    ruang: pd.DataFrame
    kelas: pd.DataFrame
    jam_slot: list                      # label jam tiap slot dalam sehari
    n_hari: int = len(HARI)
    # --- hasil pra-proses (diisi di __post_init__) ---
    n_sesi: int = field(init=False)
    n_slot: int = field(init=False)     # total slot = hari x slot per hari
    n_ruang: int = field(init=False)
    dosen_idx: np.ndarray = field(init=False)     # indeks dosen tiap sesi
    kelas_idx: np.ndarray = field(init=False)     # indeks kelas tiap sesi
    jumlah_mhs: np.ndarray = field(init=False)    # jumlah mahasiswa tiap sesi
    kapasitas: np.ndarray = field(init=False)     # kapasitas tiap ruang
    tidak_disukai: np.ndarray = field(init=False) # matriks [dosen x slot] bernilai 1 jika tidak disukai

    def __post_init__(self):
        self.slot_per_hari = len(self.jam_slot)
        self.n_sesi = len(self.mata_kuliah)
        self.n_slot = self.n_hari * self.slot_per_hari
        self.n_ruang = len(self.ruang)

        kode_dosen = {k: i for i, k in enumerate(self.dosen["kode"])}
        kode_kelas = {k: i for i, k in enumerate(self.kelas["kode"])}
        mhs_kelas = dict(zip(self.kelas["kode"], self.kelas["jumlah_mhs"]))

        self.dosen_idx = self.mata_kuliah["dosen"].map(kode_dosen).to_numpy()
        self.kelas_idx = self.mata_kuliah["kelas"].map(kode_kelas).to_numpy()
        self.jumlah_mhs = self.mata_kuliah["kelas"].map(mhs_kelas).to_numpy()
        self.kapasitas = self.ruang["kapasitas"].to_numpy()
        if np.isnan(self.dosen_idx.astype(float)).any() or np.isnan(self.kelas_idx.astype(float)).any():
            raise ValueError("Ada kode dosen/kelas di mata_kuliah.csv yang tidak terdaftar.")

        self.tidak_disukai = np.zeros((len(self.dosen), self.n_slot), dtype=int)
        for i, teks in enumerate(self.dosen["slot_tidak_disukai"].fillna("")):
            for item in filter(None, str(teks).split(";")):
                hari, s = item.strip().split("-")
                self.tidak_disukai[i, self.slot_ke_indeks(HARI.index(hari), int(s) - 1)] = 1

    # Slot global 0..24 <-> (hari, slot dalam hari)
    def slot_ke_indeks(self, hari: int, slot: int) -> int:
        return hari * self.slot_per_hari + slot

    def indeks_ke_slot(self, idx: int):
        return divmod(int(idx), self.slot_per_hari)


def muat_data(folder="data") -> DataJadwal:
    folder = Path(folder)
    slot = pd.read_csv(folder / "slot.csv")
    return DataJadwal(
        mata_kuliah=pd.read_csv(folder / "mata_kuliah.csv"),
        dosen=pd.read_csv(folder / "dosen.csv"),
        ruang=pd.read_csv(folder / "ruang.csv"),
        kelas=pd.read_csv(folder / "kelas.csv"),
        jam_slot=slot["jam"].tolist(),
    )
