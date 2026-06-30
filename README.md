# Bicomment

**BiComment** adalah website moderasi komentar preventif berbasis Fine-Tuning IndoBERT yang dikembangkan untuk membantu proses klasifikasi komentar berbahasa Indonesia sebelum komentar dipublikasikan.

## Fitur Utama

- Login pengguna
- Halaman utama berisi daftar konten
- Input komentar
- Moderasi komentar secara otomatis
- Klasifikasi komentar dua tahap
- Peringatan untuk komentar yang terindikasi pelanggaran
- Halaman tentang sistem

## Mekanisme Klasifikasi

Sistem menggunakan mekanisme klasifikasi dua tahap:

1. Tahap pertama:
   - Positive
   - Neutral
   - Terindikasi pelanggaran

2. Tahap kedua:
   - Insult
   - Hate Speech
   - Threat

Komentar yang termasuk kategori positive atau neutral akan ditampilkan pada website. Komentar yang terindikasi mengandung pelanggaran akan diberikan peringatan dan tidak dipublikasikan.

## Teknologi yang Digunakan

- Python
- Streamlit
- IndoBERT
- Hugging Face Transformers
- PyTorch
- Pandas
- Scikit-learn

## Struktur Repository

```text
BiComment/
├── app.py
├── style.css
├── requirements.txt
├── komentar.csv
├── preprocessing_data.ipynb
├── training_model.ipynb
├── README.md
├── LICENSE
└── .gitignore
