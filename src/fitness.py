"""Fungsi fitness dan pengecekan kendala penjadwalan kuliah.

Representasi (sama dengan operators.py):
    kromosom : array (n_sesi, 2); gen ke-i = (slot_global, indeks_ruang) untuk sesi kuliah ke-i
               (urutan sesi = urutan baris mata_kuliah.csv).
    populasi : array (N, n_sesi, 2).

Kendala keras (menentukan fitness). Tiap kendala dihitung sebagai jumlah "kelebihan" pada
setiap kelompok yang bentrok: jika k sesi menempati dosen/ruang/kelas yang sama pada slot
yang sama, pelanggarannya k - 1.
    1. dosen_bentrok : satu dosen mengajar di dua tempat pada slot yang sama
    2. ruang_bentrok : satu ruang dipakai dua kelas pada slot yang sama
    3. kelas_bentrok : satu kelas mahasiswa punya dua sesi pada slot yang sama
    4. kapasitas     : jumlah mahasiswa melebihi kapasitas ruang (dihitung per sesi)

    fitness = 1 / (1 + total pelanggaran keras)   -> 1.0 berarti jadwal valid

Kendala lunak (UTS: hanya dilaporkan, belum masuk fitness):
    - preferensi_dosen : jumlah sesi yang jatuh pada slot yang tidak disukai dosen
    - jeda_panjang     : jumlah slot kosong pada jeda panjang (>= JEDA_PANJANG slot kosong
                         berturut-turut di antara dua sesi kelas yang sama pada hari yang sama)
"""
import numpy as np

from .data_loader import DataJadwal

JEDA_PANJANG = 2   # jeda dianggap panjang jika ada >= 2 slot kosong berturut-turut
NAMA_KERAS = ("dosen_bentrok", "ruang_bentrok", "kelas_bentrok", "kapasitas")


# ------------------------------------------------------------------ validasi
def validasi_kromosom(krom, data: DataJadwal):
    """Pastikan bentuk, tipe, dan rentang nilai kromosom/populasi sesuai data. Raise ValueError jika tidak."""
    a = np.asarray(krom)
    if a.ndim not in (2, 3) or a.shape[-2:] != (data.n_sesi, 2):
        raise ValueError(f"Bentuk kromosom harus (n_sesi={data.n_sesi}, 2) atau (N, {data.n_sesi}, 2); "
                         f"diterima {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Kromosom harus bertipe integer; diterima {a.dtype}")
    if a.size:
        if a[..., 0].min() < 0 or a[..., 0].max() >= data.n_slot:
            raise ValueError(f"Slot di luar rentang 0..{data.n_slot - 1}")
        if a[..., 1].min() < 0 or a[..., 1].max() >= data.n_ruang:
            raise ValueError(f"Ruang di luar rentang 0..{data.n_ruang - 1}")
    return a


def _ke_populasi(krom, data):
    a = validasi_kromosom(krom, data)
    return (a[None], True) if a.ndim == 2 else (a, False)


# ------------------------------------------------------------------ inti perhitungan (vektor)
def _hitung_bentrok(slot, entitas, n_entitas, n_slot):
    """Jumlah kelebihan sesi per (entitas, slot) untuk tiap individu.

    slot    : (N, S) slot tiap sesi
    entitas : (S,) atau (N, S) indeks dosen/ruang/kelas tiap sesi
    return  : (N,) jumlah pelanggaran
    """
    N = slot.shape[0]
    ukuran = n_entitas * n_slot
    kunci = entitas * n_slot + slot + np.arange(N)[:, None] * ukuran
    cacah = np.bincount(kunci.ravel(), minlength=N * ukuran).reshape(N, ukuran)
    return np.maximum(cacah - 1, 0).sum(axis=1)


def _hitung_keras(pop, data):
    slot, ruang = pop[:, :, 0], pop[:, :, 1]
    return {
        "dosen_bentrok": _hitung_bentrok(slot, data.dosen_idx[None, :], len(data.dosen), data.n_slot),
        "ruang_bentrok": _hitung_bentrok(slot, ruang, data.n_ruang, data.n_slot),
        "kelas_bentrok": _hitung_bentrok(slot, data.kelas_idx[None, :], len(data.kelas), data.n_slot),
        "kapasitas": (data.jumlah_mhs[None, :] > data.kapasitas[ruang]).sum(axis=1),
    }


# ------------------------------------------------------------------ API utama (dipakai ga.py)
def total_pelanggaran(krom, data: DataJadwal):
    """Total pelanggaran kendala keras.

    return (total, rincian). Untuk satu kromosom: (int, {nama: int}).
    Untuk populasi (N, S, 2): (array (N,), {nama: array (N,)}).
    """
    pop, tunggal = _ke_populasi(krom, data)
    rinci = _hitung_keras(pop, data)
    total = sum(rinci.values())
    if tunggal:
        return int(total[0]), {k: int(v[0]) for k, v in rinci.items()}
    return total, rinci


def fitness(krom, data: DataJadwal):
    """fitness = 1 / (1 + total pelanggaran keras). Populasi -> array (N,), satu kromosom -> float."""
    total, _ = total_pelanggaran(krom, data)
    hasil = 1.0 / (1.0 + np.asarray(total, dtype=float))
    return float(hasil) if np.ndim(hasil) == 0 else hasil


# ------------------------------------------------------------------ pengecekan bentrok (detail)
def _kelompok_bentrok(slot, entitas):
    """Kelompok sesi yang berbagi (entitas, slot): list of (entitas, slot, [indeks sesi])."""
    kelompok = {}
    for i, (e, s) in enumerate(zip(entitas, slot)):
        kelompok.setdefault((int(e), int(s)), []).append(i)
    return [(e, s, idx) for (e, s), idx in sorted(kelompok.items()) if len(idx) > 1]


def pelanggaran_keras(krom, data: DataJadwal) -> dict:
    """Rincian kendala keras untuk SATU kromosom: {nama: (jumlah, detail)}.

    detail: untuk bentrok = list (indeks_entitas, slot, [indeks sesi]); untuk kapasitas =
    list (indeks sesi, indeks ruang, jumlah_mhs, kapasitas).
    """
    krom = validasi_kromosom(krom, data)
    if krom.ndim != 2:
        raise ValueError("pelanggaran_keras hanya menerima satu kromosom")
    slot, ruang = krom[:, 0], krom[:, 1]
    hasil = {}
    for nama, entitas in (("dosen_bentrok", data.dosen_idx),
                          ("ruang_bentrok", ruang),
                          ("kelas_bentrok", data.kelas_idx)):
        grup = _kelompok_bentrok(slot, entitas)
        hasil[nama] = (sum(len(idx) - 1 for _, _, idx in grup), grup)
    lebih = [(i, int(ruang[i]), int(data.jumlah_mhs[i]), int(data.kapasitas[ruang[i]]))
             for i in range(data.n_sesi) if data.jumlah_mhs[i] > data.kapasitas[ruang[i]]]
    hasil["kapasitas"] = (len(lebih), lebih)
    return hasil


def jelaskan_pelanggaran(krom, data: DataJadwal) -> list:
    """Daftar pelanggaran keras dalam kalimat yang mudah dibaca. List kosong = jadwal valid."""
    from .data_loader import HARI
    pesan = []
    nama_entitas = {"dosen_bentrok": ("Dosen", data.dosen["nama"]),
                    "ruang_bentrok": ("Ruang", data.ruang["kode"]),
                    "kelas_bentrok": ("Kelas", data.kelas["kode"])}
    rinci = pelanggaran_keras(krom, data)
    for nama, (label, daftar) in nama_entitas.items():
        for e, s, idx in rinci[nama][1]:
            h, sl = data.indeks_ke_slot(s)
            mk = ", ".join(str(data.mata_kuliah["kode"].iloc[i]) for i in idx)
            pesan.append(f"{label} {daftar.iloc[e]} bentrok pada {HARI[h]} slot {sl + 1}: {mk}")
    for i, r, mhs, kap in rinci["kapasitas"][1]:
        pesan.append(f"Kapasitas: {data.mata_kuliah['kode'].iloc[i]} ({mhs} mhs) "
                     f"di ruang {data.ruang['kode'].iloc[r]} (kapasitas {kap})")
    return pesan


# ------------------------------------------------------------------ kendala lunak (dilaporkan saja)
def _jeda_panjang(krom, data):
    total = 0
    for k in range(len(data.kelas)):
        slot = np.unique(krom[data.kelas_idx == k, 0])
        hari = slot // data.slot_per_hari
        for h in np.unique(hari):
            kosong = np.diff(np.sort(slot[hari == h])) - 1
            total += int(kosong[kosong >= JEDA_PANJANG].sum())
    return total


def pelanggaran_lunak(krom, data: DataJadwal) -> dict:
    """Kendala lunak untuk SATU kromosom: {'preferensi_dosen': int, 'jeda_panjang': int}."""
    krom = validasi_kromosom(krom, data)
    if krom.ndim != 2:
        raise ValueError("pelanggaran_lunak hanya menerima satu kromosom")
    return {
        "preferensi_dosen": int(data.tidak_disukai[data.dosen_idx, krom[:, 0]].sum()),
        "jeda_panjang": _jeda_panjang(krom, data),
    }
