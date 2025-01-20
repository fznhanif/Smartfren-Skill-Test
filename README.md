Berikut adalah user guide:

- Telah dikerjakan test skill untuk Smartfren atas nama kandidat Fauzan Hanif pelamar sebagai Data Scientist.
- Dokumentasi ini menjelaskan struktur direktori utama dan panduan penggunaan fokus pada file pipeline.py dan aplikasi berbasis Streamlit yaitu app.py.

Smartfren-Skill-Test/
├── src/
│   ├── pipeline.py  # Logika pemrosesan data utama
│   ├── app.py       # Aplikasi berbasis Streamlit untuk antarmuka pengguna
├── data/
│   ├── Supply Chain Management_train.csv   # Dataset untuk pelatihan model
│   ├── Supply Chain Management_test.csv    # Dataset untuk pengujian model
├── models/
│   ├── pca_model.pkl           # Model PCA yang disimpan
│   ├── multi_target_model.pkl  # Model multitarget yang disimpan
├── notebook/
│   ├── 01_EDA.ipynb      # Notebook Jupyter untuk analisis dan eksperimen
│   ├── 02_PCA.ipynb      # Notebook Jupyter untuk analisis dan eksperimen
│   ├── 03_Modeling.ipynb # Notebook Jupyter untuk analisis dan eksperimen
├── requirements.txt      # Daftar dependensi Python
├── README.md             # Dokumentasi proyek

- Cara Penggunaan:
>  pipeline.py
File pipeline.py bertanggung jawab untuk:
- Mengolah data mentah menjadi format yang siap digunakan oleh aplikasi.
- Mengintegrasikan berbagai sumber data jika diperlukan.
- Menyediakan fungsi atau kelas yang dapat diimpor ke aplikasi utama.
> app.py
File app.py adalah aplikasi berbasis Streamlit yang menyediakan antarmuka pengguna interaktif.
