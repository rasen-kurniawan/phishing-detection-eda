# 📩🔐 EDA Dashboard — Deteksi Phishing SMS Indonesia

Interactive Exploratory Data Analysis (EDA) dashboard for Indonesian phishing SMS detection. Analyzes 1,772 messages across 5 business questions using statistical and NLP-based approaches. Built with Streamlit, Matplotlib, and Scikit-learn.

---

## 📁 Struktur Folder

```
├── Dashboard_baru.py                 # Script utama Streamlit
├── dataset_phishing_sms_indo.csv     # Dataset phishing SMS Indonesia
├── requirements.txt                  # Daftar dependencies
└── README.md                         # Dokumentasi proyek
```

---

## ⚙️ Setup Environment (Anaconda)

### 1. Buat virtual environment baru

```bash
conda create -n phishing-eda python=3.10
```

### 2. Aktifkan environment

```bash
conda activate phishing-eda
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Setup Environment (Shell / Terminal)

### 1. Buat virtual environment

```bash
python -m venv venv
```

### 2. Aktifkan environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Dashboard

Pastikan environment sudah aktif dan berada di folder project, lalu jalankan:

```bash
streamlit run Dashboard_baru.py
```

Dashboard akan otomatis terbuka di browser pada `http://localhost:8501`

---

## 📊 Isi Dashboard

Dashboard terdiri dari 6 halaman yang dapat dipilih via sidebar:

| Halaman | Deskripsi |
|---|---|
| Overview & Statistik Deskriptif | Statistik dasar, distribusi sumber, preview data per kelas |
| BQ1 · Distribusi Kelas | Proporsi phishing vs normal, imbalance ratio |
| BQ2 · Modus Phishing | Taktik dan kombinasi taktik yang paling dominan |
| BQ3 · Fitur Struktural | Korelasi URL, nomor telepon, dan fitur lain terhadap label |
| BQ4 · Kata & Frasa | Top kata, TF-IDF, dan Word Cloud per kelas |
| BQ5 · Panjang Teks | Distribusi panjang teks dan uji statistik Mann-Whitney U |
