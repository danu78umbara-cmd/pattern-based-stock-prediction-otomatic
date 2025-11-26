# 📈 Prediksi Harga Emas dengan Flask API & Yahoo Finance

Repositori ini berisi proyek end-to-end untuk prediksi arah harga emas jangka pendek menggunakan kombinasi pola candlestick, indikator teknikal, dan model Machine Learning berbasis ensemble.

Keunikan proyek ini adalah integrasi antara:

- Web API Flask
- Pengambilan data real-time dari Yahoo Finance (yfinance)
- Pemilihan model otomatis berdasarkan pola candlestick terbaru
- Antarmuka HTML untuk input dan visualisasi hasil

Proyek ini berfungsi sebagai layanan prediksi yang dapat digunakan untuk aplikasi web, otomasi trading sederhana, maupun pembelajaran market analytics.

------------------------------------------------------------

# 🚀 Alur Kerja Proyek (End-to-End)

```
[User Input] → [Fetch Data via yfinance] → [Feature Engineering] → 
[Deteksi Pola Candlestick] → [Load Model Sesuai Pola] →
[Prediksi Naik/Turun] → [Tampilkan Hasil di HTML]
```

------------------------------------------------------------

1. Pelatihan Model (Offline, Opsional)

- Data historis harga emas diambil dari Yahoo Finance.
- Data dibersihkan dan diekstraksi indikator teknikal.
- Pola candlestick diklasifikasikan.
- Untuk setiap pola, dilatih satu model Machine Learning berbeda.
- Model terbaik disimpan dalam folder saved_models/ sebagai file .pkl.

Catatan: Training tidak harus diulang jika model sudah tersedia.

------------------------------------------------------------

2. Penerapan API (Online dengan Flask)

Dilakukan oleh file app.py:

- API berjalan menggunakan Flask.
- Saat startup, Flask memuat semua file model .pkl.
- User memasukkan ticker emas (misal: XAUUSD).
- Sistem mengambil harga terkini menggunakan yfinance.
- Data diproses oleh feature_engineering() dari preprocessor.py.
- Pola candlestick terakhir dianalisis.
- Sistem memilih model sesuai pola (misalnya model_Hammer.pkl).
- Model menghasilkan prediksi:
    1 = Naik
    0 = Turun
- Hasil dikembalikan ke halaman HTML melalui templates/index.html.

------------------------------------------------------------

## 🗂️ Struktur Repositori
```
.
├── 📁 data_emas/           # (Tempat data CSV mentah Anda)
├── 📁 saved_models/        # (Tempat model .pkl disimpan)
├── 📁 templates/           # (Berisi file HTML untuk antarmuka Flask)
│
├── 📜 app.py               # (Aplikasi FASK utama untuk API/Web)
├── 📜 main_workflow        # (Notebook untuk analisis, preprocessing & training)
├── 📜 preprocessor.py      # (Modul .py untuk fungsi preprocessing)
├── 📜 README.md            # (Dokumentasi ini)
```

# 🛠️ Cara Menjalankan Sistem

1. Jalankan Flask API
```bash
python app.py
```
Akses melalui browser:
`http://127.0.0.1:5000`

2. Proses Prediksi di Frontend

Pada halaman HTML:

- Masukkan ticker (misal XAUUSD)
- Klik Submit

Backend otomatis:

- Mengambil data terbaru via yfinance
- Melakukan preprocessing
- Mendeteksi pola candlestick terbaru
- Memilih model .pkl sesuai pola
- Menampilkan hasil prediksi:
    - Naik / Turun
    - Pola candlestick terakhir
    - Ringkasan data

# 💡 Teknologi yang Digunakan

Machine Learning:
- Scikit-learn
- Random Forest
- Logistic Regression
- Pola candlestick
- Teknik balancing data (SMOTE, optional)

Backend & API
- Flask
- joblib
- yfinance

Frontend
- HTML (templates/index.html)
- CSS sederhana

# 📌 Catatan Tambahan

- Sistem membaca pola candlestick terakhir untuk menentukan model mana yang digunakan.
- Jika model untuk pola tersebut tidak ditemukan, API akan memakai model default.
- Data real-time dari Yahoo Finance memastikan prediksi selalu menggunakan data terbaru.