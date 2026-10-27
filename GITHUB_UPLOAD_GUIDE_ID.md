# Panduan unggah repositori EIS ke GitHub

Repositori ini merupakan paket **anonim untuk penelaahan naskah**. Tidak ada perubahan pada kode atau hasil penelitian; README dan file pendukung GitHub ditambahkan untuk keterulangan (*reproducibility*).

## 1. Persiapan

1. Ekstrak `EIS_GitHub_Ready.zip` sehingga muncul folder `eis-identifiability-code/`.
2. Buat repositori **kosong** di GitHub (jangan pilih pilihan otomatis membuat README, `.gitignore`, atau lisensi, karena file tersebut sudah tersedia).
3. Jika masih mengikuti proses **double-blind review**, jangan gunakan akun GitHub yang menampilkan identitas penulis. Periksa juga nama dan surel pembuat *commit* serta riwayat publik, sesuai kebijakan jurnal. Mode private dapat membatasi akses reviewer; koordinasikan cara akses melalui prosedur jurnal.
4. Jangan unggah file ZIP sebagai **satu-satunya** isi repositori karena GitHub tidak akan menampilkan dan menjalankan struktur kode di dalam ZIP. Unggah **isi folder** `eis-identifiability-code/`.

## 2. Cara unggah (disarankan: Git di komputer)

Buka Terminal / Git Bash di dalam folder `eis-identifiability-code/`, lalu:

```bash
git init
git add .
git commit -m "Initial anonymous EIS reproducibility release"
git branch -M main
git remote add origin https://github.com/ACCOUNT/REPOSITORY.git
git push -u origin main
```

Ganti `ACCOUNT` dan `REPOSITORY` dengan identitas repositori yang telah dibuat. Jika Git meminta pengaturan nama dan email penulis commit, gunakan konfigurasi yang sesuai dengan persyaratan anonimitas dan kebijakan akun Anda. **Jangan menggunakan identitas palsu untuk menyesatkan reviewer; gunakan mekanisme blinded sharing yang diizinkan jurnal.**

Mengunggah melalui browser GitHub memungkinkan untuk jumlah file terbatas, tetapi paket ini memiliki ratusan file dan beberapa direktori; Git lebih praktis.

## 3. Pengecekan sesudah unggah

Jalankan pengecekan struktur sebelum *push*:

```bash
python scripts/validate_repository.py
```

Setelah upload, buka halaman GitHub dan pastikan `README.md`, `PreparedData.zip`, `src/`, `eisvarpro/`, `results/`, `manuscript_tables/` dan `manuscript_figures/` tersedia. Periksa *Actions*; workflow yang disediakan memeriksa integritas arsip, **bukan** menjalankan seluruh studi numerik.

## 4. Catatan reprodusibilitas ilmiah

- Uji pustaka `eisvarpro` secara terpisah dengan `python -m pytest -q tests` dari direktori `eisvarpro/` setelah menginstal dependensi.
- Analisis lengkap membutuhkan waktu dan akan menghasilkan file baru/menimpa beberapa hasil; lihat `README.md` dan `run_all.sh`.
- Data awal berasal dari Falagario et al. (2026), doi:10.1038/s41597-026-07652-y, dengan arsip Zenodo doi:10.5281/zenodo.18769577 berlisensi CC BY 4.0.
- Nama dan nomor tabel di dalam kode dapat berbeda dari versi PDF manuskrip tertentu. Pastikan kembali sebelum menyitir nomor tabel.
- Jika jurnal meminta *Code availability statement*, hanya cantumkan URL setelah benar-benar dapat diakses reviewer.
