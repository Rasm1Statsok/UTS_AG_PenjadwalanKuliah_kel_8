"""Unit test untuk src/fitness.py (pengecekan bentrok dan fungsi fitness).

Jalankan dari root repositori:
    python -m pytest tests/ -v
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import operators as op
from src.data_loader import DataJadwal, muat_data
from src.fitness import (fitness, jelaskan_pelanggaran, pelanggaran_keras,
                         pelanggaran_lunak, total_pelanggaran, validasi_kromosom)
from src.ga import KonfigurasiAG, jalankan_ag

FOLDER_DATA = Path(__file__).resolve().parent.parent / "data"
JAM = ["07.00-08.40", "08.40-10.20", "10.20-12.00", "12.50-14.30", "14.30-16.10"]


# ----------------------------------------------------------------- data mini untuk uji terkontrol
def buat_data(sesi, tidak_disukai=None):
    """Data kecil: 3 dosen (D1-D3), 3 kelas (K1 35, K2 35, K3 25 mhs), 3 ruang (R1 40, R2 40, R3 30).

    sesi: list of (kode_dosen, kode_kelas), satu elemen per sesi kuliah.
    """
    tidak_disukai = tidak_disukai or {}
    return DataJadwal(
        mata_kuliah=pd.DataFrame([{"kode": f"MK{i}", "nama": f"Matkul {i}", "sks": 3, "dosen": d, "kelas": k}
                                  for i, (d, k) in enumerate(sesi)]),
        dosen=pd.DataFrame({"kode": ["D1", "D2", "D3"], "nama": ["A", "B", "C"],
                            "slot_tidak_disukai": [tidak_disukai.get(d, "") for d in ("D1", "D2", "D3")]}),
        ruang=pd.DataFrame({"kode": ["R1", "R2", "R3"], "kapasitas": [40, 40, 30]}),
        kelas=pd.DataFrame({"kode": ["K1", "K2", "K3"], "jumlah_mhs": [35, 35, 25]}),
        jam_slot=JAM,
    )


SESI = [("D1", "K1"), ("D2", "K2"), ("D3", "K3"), ("D1", "K2"), ("D2", "K3"), ("D3", "K1")]


@pytest.fixture
def data():
    return buat_data(SESI)


@pytest.fixture
def valid():
    """Jadwal tanpa bentrok: sesi ke-i di slot i, semuanya di ruang R1."""
    return np.array([[i, 0] for i in range(len(SESI))])


# ----------------------------------------------------------------- kendala keras satu per satu
def test_jadwal_valid_tanpa_pelanggaran(data, valid):
    total, rinci = total_pelanggaran(valid, data)
    assert total == 0
    assert all(v == 0 for v in rinci.values())
    assert fitness(valid, data) == 1.0
    assert jelaskan_pelanggaran(valid, data) == []


def test_dosen_bentrok(data, valid):
    k = valid.copy()
    k[3] = [0, 1]            # D1 mengajar sesi 0 dan 3 di slot 0, beda ruang & beda kelas
    total, rinci = total_pelanggaran(k, data)
    assert rinci == {"dosen_bentrok": 1, "ruang_bentrok": 0, "kelas_bentrok": 0, "kapasitas": 0}
    assert total == 1


def test_ruang_bentrok(data, valid):
    k = valid.copy()
    k[1] = [0, 0]            # sesi 0 dan 1 sama-sama di slot 0 ruang R1 (dosen & kelas beda)
    _, rinci = total_pelanggaran(k, data)
    assert rinci == {"dosen_bentrok": 0, "ruang_bentrok": 1, "kelas_bentrok": 0, "kapasitas": 0}


def test_kelas_bentrok(data, valid):
    k = valid.copy()
    k[5] = [0, 1]            # K1 punya sesi 0 dan 5 di slot 0 (beda dosen, beda ruang)
    _, rinci = total_pelanggaran(k, data)
    assert rinci == {"dosen_bentrok": 0, "ruang_bentrok": 0, "kelas_bentrok": 1, "kapasitas": 0}


def test_kapasitas_terlampaui(data, valid):
    k = valid.copy()
    k[0] = [0, 2]            # K1 (35 mhs) di R3 (kapasitas 30)
    _, rinci = total_pelanggaran(k, data)
    assert rinci == {"dosen_bentrok": 0, "ruang_bentrok": 0, "kelas_bentrok": 0, "kapasitas": 1}


def test_kapasitas_pas_tidak_dihitung():
    # K3 (25 mhs) di R3 (30) aman; K1 (35 mhs) di R1 (40) aman; batas sama persis (35 vs 35) juga aman
    d = buat_data([("D1", "K1")])
    d.kapasitas = np.array([35, 40, 30])
    assert total_pelanggaran(np.array([[0, 0]]), d)[0] == 0


def test_tiga_sesi_dosen_sama_dihitung_dua():
    d = buat_data([("D1", "K1"), ("D1", "K2"), ("D1", "K3")])
    k = np.array([[0, 0], [0, 1], [0, 2]])    # K1->R1, K2->R2, K3->R3: hanya dosen yang bentrok
    _, rinci = total_pelanggaran(k, d)
    assert rinci["dosen_bentrok"] == 2
    assert sum(rinci.values()) == 2


def test_slot_berbeda_hari_tidak_bentrok(data, valid):
    k = valid.copy()
    k[3] = [5, 0]            # D1: sesi 0 di Senin slot 1, sesi 3 di Selasa slot 1, ruang R1 sama
    k[5] = [3, 0]            # (sesi 5 dipindah agar tidak menempati slot 5 di R1)
    assert total_pelanggaran(k, data)[0] == 0


def test_slot_sama_ruang_berbeda_tanpa_bentrok(data, valid):
    k = valid.copy()
    k[1] = [0, 1]            # slot sama dengan sesi 0 tapi dosen, kelas, ruang semuanya beda
    assert total_pelanggaran(k, data)[0] == 0


# ----------------------------------------------------------------- fungsi fitness
def test_rumus_fitness(data, valid):
    k = valid.copy()
    k[3] = [0, 1]            # 1 dosen_bentrok
    k[1] = [1, 2]            # K2 (35 mhs) di R3 (30) -> 1 kapasitas; slot 1 tidak dipakai sesi lain di R3
    total, _ = total_pelanggaran(k, data)
    assert total == 2
    assert fitness(k, data) == pytest.approx(1 / 3)


def test_fitness_lebih_baik_jika_pelanggaran_lebih_sedikit(data, valid):
    satu = valid.copy(); satu[3] = [0, 1]
    dua = satu.copy(); dua[5] = [0, 2]       # tambah: K1 bentrok (slot 0) + kapasitas K1 di R3
    assert fitness(valid, data) > fitness(satu, data) > fitness(dua, data)


def test_fitness_selalu_dalam_rentang():
    d = muat_data(FOLDER_DATA)
    pop = op.inisialisasi_populasi(200, d.n_sesi, d.n_slot, d.n_ruang, np.random.default_rng(0))
    f = fitness(pop, d)
    assert f.shape == (200,)
    assert np.all(f > 0) and np.all(f <= 1)


def test_kendala_lunak_tidak_memengaruhi_fitness():
    d = buat_data(SESI, tidak_disukai={"D1": "Senin-1"})
    k = np.array([[i, 0] for i in range(len(SESI))])     # sesi 0 (D1) di Senin-1, tanpa bentrok keras
    assert pelanggaran_lunak(k, d)["preferensi_dosen"] == 1
    assert fitness(k, d) == 1.0


# ----------------------------------------------------------------- konsistensi dengan hitungan naif
def hitung_naif(krom, d):
    """Pembanding sederhana (loop + Counter) untuk memastikan versi vektor benar."""
    krom = [tuple(int(x) for x in g) for g in krom]
    def kelebihan(kunci):
        return sum(max(c - 1, 0) for c in Counter(kunci).values())
    return {
        "dosen_bentrok": kelebihan((int(d.dosen_idx[i]), s) for i, (s, r) in enumerate(krom)),
        "ruang_bentrok": kelebihan((r, s) for s, r in krom),
        "kelas_bentrok": kelebihan((int(d.kelas_idx[i]), s) for i, (s, r) in enumerate(krom)),
        "kapasitas": sum(int(d.jumlah_mhs[i]) > int(d.kapasitas[r]) for i, (s, r) in enumerate(krom)),
    }


def test_versi_vektor_sama_dengan_hitungan_naif_dan_per_individu():
    d = muat_data(FOLDER_DATA)
    pop = op.inisialisasi_populasi(60, d.n_sesi, d.n_slot, d.n_ruang, np.random.default_rng(123))
    total_pop, rinci_pop = total_pelanggaran(pop, d)
    f_pop = fitness(pop, d)
    for i, krom in enumerate(pop):
        naif = hitung_naif(krom, d)
        total_i, rinci_i = total_pelanggaran(krom, d)
        assert rinci_i == naif
        assert {k: int(v[i]) for k, v in rinci_pop.items()} == naif
        assert total_i == total_pop[i] == sum(naif.values())
        assert fitness(krom, d) == pytest.approx(f_pop[i])


def test_pelanggaran_keras_detail_konsisten_dengan_total():
    d = muat_data(FOLDER_DATA)
    pop = op.inisialisasi_populasi(30, d.n_sesi, d.n_slot, d.n_ruang, np.random.default_rng(7))
    for krom in pop:
        rinci = pelanggaran_keras(krom, d)
        assert {k: v[0] for k, v in rinci.items()} == total_pelanggaran(krom, d)[1]
        assert len(jelaskan_pelanggaran(krom, d)) == sum(len(v[1]) for v in rinci.values())


def test_jelaskan_pelanggaran_menyebut_sumber_bentrok(data, valid):
    k = valid.copy()
    k[3] = [0, 1]
    pesan = jelaskan_pelanggaran(k, data)
    assert len(pesan) == 1
    assert "Dosen A" in pesan[0] and "Senin slot 1" in pesan[0] and "MK0" in pesan[0] and "MK3" in pesan[0]


# ----------------------------------------------------------------- kendala lunak
def test_preferensi_dosen_dihitung_per_sesi():
    d = buat_data(SESI, tidak_disukai={"D1": "Senin-1;Selasa-2"})   # slot global 0 dan 6
    k = np.array([[0, 0], [1, 0], [2, 0], [6, 0], [3, 1], [4, 1]])  # sesi 0 & 3 (D1) kena
    assert pelanggaran_lunak(k, d)["preferensi_dosen"] == 2


def test_jeda_panjang_dihitung_dari_slot_kosong_berturut():
    d = buat_data(SESI)
    k = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [3, 1]])
    # K1 (sesi 0 & 5) di slot 0 dan 3 -> 2 slot kosong = jeda panjang (2); K2 slot 1,3 dan K3 slot 2,4 -> 1 kosong
    assert pelanggaran_lunak(k, d)["jeda_panjang"] == 2


def test_jeda_satu_slot_dan_beda_hari_bukan_jeda_panjang():
    d = buat_data(SESI)
    k = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [2, 1]])  # K1 di slot 0 & 2 -> hanya 1 kosong
    assert pelanggaran_lunak(k, d)["jeda_panjang"] == 0
    k[5] = [9, 1]                                                   # K1 di Senin & Selasa -> beda hari
    assert pelanggaran_lunak(k, d)["jeda_panjang"] == 0


def test_kunci_lunak_sesuai_yang_dipakai_eksperimen(data, valid):
    assert set(pelanggaran_lunak(valid, data)) == {"preferensi_dosen", "jeda_panjang"}


# ----------------------------------------------------------------- validasi input
def test_slot_di_luar_rentang_ditolak(data, valid):
    k = valid.copy(); k[0, 0] = data.n_slot
    with pytest.raises(ValueError):
        fitness(k, data)
    k[0, 0] = -1
    with pytest.raises(ValueError):
        validasi_kromosom(k, data)


def test_ruang_di_luar_rentang_ditolak(data, valid):
    k = valid.copy(); k[0, 1] = data.n_ruang
    with pytest.raises(ValueError):
        total_pelanggaran(k, data)


def test_bentuk_atau_tipe_salah_ditolak(data, valid):
    with pytest.raises(ValueError):
        fitness(valid[:-1], data)                 # jumlah sesi kurang
    with pytest.raises(ValueError):
        fitness(valid.astype(float), data)        # bukan integer
    with pytest.raises(ValueError):
        pelanggaran_lunak(valid[None], data)      # kendala lunak hanya untuk satu kromosom


# ----------------------------------------------------------------- data UTS & integrasi dengan AG
def test_data_uts_layak_secara_dasar():
    d = muat_data(FOLDER_DATA)
    assert d.n_sesi == 20 and len(d.dosen) == 8 and d.n_ruang == 5 and d.n_slot == 25
    assert (d.jumlah_mhs <= d.kapasitas.max()).all()                  # tiap sesi muat di minimal satu ruang
    assert np.bincount(d.dosen_idx).max() <= d.n_slot                 # beban dosen tidak melebihi slot
    assert np.bincount(d.kelas_idx).max() <= d.n_slot                 # beban kelas tidak melebihi slot
    assert d.n_sesi <= d.n_slot * d.n_ruang


def test_hasil_ag_konsisten_dengan_fitness():
    d = muat_data(FOLDER_DATA)
    h = jalankan_ag(d, KonfigurasiAG(ukuran_populasi=40, maks_generasi=30, seed=5))
    total, _ = total_pelanggaran(h.terbaik, d)
    assert h.pelanggaran_terbaik == total
    assert h.fitness_terbaik == pytest.approx(1 / (1 + total))


def test_ag_menemukan_jadwal_valid_pada_data_uts():
    d = muat_data(FOLDER_DATA)
    h = jalankan_ag(d, KonfigurasiAG(ukuran_populasi=100, maks_generasi=100, seed=42))
    assert h.generasi_valid_pertama is not None
    assert h.fitness_terbaik == 1.0
    assert total_pelanggaran(h.terbaik, d)[0] == 0
    assert jelaskan_pelanggaran(h.terbaik, d) == []
