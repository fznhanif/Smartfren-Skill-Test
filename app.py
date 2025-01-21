import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Menentukan direktori root proyek
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data.load_data import load_arff_data

# Title aplikasi
st.title("Model Evaluasi Supply Chain Management")

# Sidebar untuk navigasi
st.sidebar.title("Menu")
menu = st.sidebar.radio("", options=["Home", "Analisis Data", "Prediksi & Visualisasi"])

# Fungsi memuat data
@st.cache_data
def load_data(file_path):
    return load_arff_data(file_path)

if menu == "Home":
    st.subheader("Selamat Datang, perkenalkan saya Fauzan Hanif")
    st.write("""
        Aplikasi ini akan memperlihatkan analisis data, 
        prediksi, evaluasi model, dan visualisasi hasil prediksi pada 
        proyek Supply Chain Management.
    """)
    st.write("""
        pada projek ini terdapat limitasi yaitu 1) hanya fokus dengan 1 model machine learning saja. 2) explorasi tidak begitu mendalam. 3) features selection baiknya dilakukan dengan pembuatan model lasso agar mendapat feature yang penting guna interpretasi dan ke-general-an model
    """)

elif menu == "Analisis Data":
    st.subheader("Analisis Data")

    # Path ke data train
    TRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data", "Supply Chain Management_train.arff")

    # Memuat data train
    st.write("Memuat data train...")
    train_data = load_data(TRAIN_DATA_PATH)
    if train_data is None or train_data.empty:
        st.error("Gagal memuat data. Periksa file data Anda.")
    else:
        st.success("Data train berhasil dimuat!")
        
        feature_columns = [col for col in train_data.columns if not (col.startswith("MTLp") or col == "LBL")]
        target_columns = [col for col in train_data.columns if col.startswith("MTLp") or col == "LBL"]
        st.text(f"data mempunyai {len(train_data)} rows dan {len(train_data.columns)} columns, indikasi data high dimension")
        st.text(f"total features = {len(feature_columns)}")
        st.text(f"total target features = {len(target_columns)}")
        st.text(f"pada data train memiliki {train_data.isnull().sum().sum()} missing values, indikasi tidak perlu imputing missing values")
        
        # Menampilkan ringkasan statistik
        st.subheader("Ringkasan Statistik")
        multiselected_col = st.multiselect("Pilih Feature yang ingin dilihat ringkasan statistiknya", train_data.columns)
        if multiselected_col:
            st.write(train_data[multiselected_col].describe())
        
        # Distribusi setiap target feature
        target_columns = [col for col in train_data.columns if col.startswith("MTLp") or col == "LBL"]
        selected_target = st.selectbox("Pilih Target Feature untuk Visualisasi Distribusi:", target_columns)

        if selected_target:
            st.subheader(f"Distribusi {selected_target}")
            fig, ax = plt.subplots()
            sns.histplot(train_data[selected_target], bins=30, kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribusi {selected_target}")
            ax.set_xlabel("Nilai")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

        # Korelasi antar fitur
        st.subheader("Matriks Korelasi Target Feature")
        corr_matrix = train_data[target_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Matriks Korelasi Target Feature")
        st.pyplot(fig)
        st.text("Dari heatmap korelasi antar feature target di atas kita mengetahui adanya korelasi antar target")

        PCA_MODEL_PATH = os.path.join(ROOT_DIR, "models", "pca_model.pkl")
        
        # Analisis PCA
        st.subheader("Analisis PCA")
        non_target_features = [col for col in train_data.columns if col not in target_columns]

        # Standardisasi fitur non-target
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train_data[non_target_features])

        # Apply PCA
        pca = PCA(n_components=None, random_state=19)
        pca_result = pca.fit_transform(scaled_data)

        # Variance explained
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        optimal_pc = np.where(cumulative_variance > 95)[0][0] + 1
        st.write(f"Jumlah komponen utama optimal untuk menjelaskan lebih dari 95% variansi: **{optimal_pc}**")

        # Visualisasi explained variance
        st.subheader("Explained Variance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label="Individual Explained Variance")
        ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where="mid", label="Cumulative Explained Variance")
        ax.set_xlabel("Komponen Utama")
        ax.set_ylabel("Variansi Terjelaskan (%)")
        ax.set_title("Variansi Terjelaskan oleh Komponen Utama")
        ax.legend()
        st.pyplot(fig)

        # Visualisasi dua komponen utama pertama
        st.subheader("Visualisasi Dua Komponen Utama Pertama")
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, edgecolor='k')
        ax.set_xlabel("Komponen Utama 1")
        ax.set_ylabel("Komponen Utama 2")
        ax.set_title("PCA: Dua Komponen Utama Pertama")
        st.pyplot(fig)

        # Deteksi Outlier dengan Isolation Forest
        st.subheader("Deteksi Outlier dengan Isolation Forest")
        isolation_forest = IsolationForest(random_state=42, contamination=0.05, n_estimators=100)
        outliers = isolation_forest.fit_predict(pca_result[:, :10])

        # Tambahkan flag outlier ke data
        train_data['Outlier'] = (outliers == -1).astype(int)
        outlier_count = train_data['Outlier'].value_counts()
        st.write("Jumlah outlier yang terdeteksi:")
        st.write(outlier_count)

        # Visualisasi outlier pada dua komponen utama pertama
        st.subheader("Visualisasi Outlier")
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=train_data['Outlier'],
            cmap='coolwarm',
            alpha=0.7,
            edgecolor='k'
        )
        plt.colorbar(scatter, ax=ax, label="Flag Outlier (1=Outlier)")
        ax.set_xlabel("Komponen Utama 1")
        ax.set_ylabel("Komponen Utama 2")
        ax.set_title("Deteksi Outlier dengan Isolation Forest")
        st.pyplot(fig)

        st.markdown("indikasi data memiliki outlier pada feature non target, melihat hasil sebaran PC1 dan PC2 hasil PCA. penyebaran data (warna merah) mengindikasikan adanya outlier yang mungkin bisa jadi karakter data atau kesalahan pada data.")
        st.markdown("melihat sebaran variasi data cenderung berkelompok pada suatu titik maka kita akan gunakan PCA sebagai dimension reduction guna meringkas data.")
        st.markdown("selanjutnya, insight dari temuan ini adalah adanya outlier maka akan digunakan model yang bisa menanggulangi outlier seperti tree-based model maka akan langsung dicoba randomforest regression.")

        
elif menu == "Prediksi & Visualisasi":
    st.subheader("Prediksi dan Visualisasi Hasil")
    
    # Path file
    TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "Supply Chain Management_test.arff")
    PCA_MODEL_PATH = os.path.join(ROOT_DIR, "models", "pca_model.pkl")
    REGRESSION_MODEL_PATH = os.path.join(ROOT_DIR, "models", "multi_target_model.pkl")
    
    # Memuat data test
    st.write("Memuat data test...")
    df = load_data(TEST_DATA_PATH)
    if df is None or df.empty:
        st.error("Gagal memuat data. Periksa file data Anda.")
    else:
        st.success("Data berhasil dimuat!")

        # Memproses data
        feature_columns = [col for col in df.columns if not (col.startswith("MTLp") or col == "LBL")]
        target_columns = [col for col in df.columns if col.startswith("MTLp") or col == "LBL"]

        X_test = df[feature_columns].astype(float)
        Y_test = df[target_columns].astype(float)

        # Memuat model
        st.write("Memuat model...")
        from src.models.load_pca_model import load_pca_model, apply_pca_model
        from src.models.load_model import load_model

        @st.cache_resource
        def load_models(pca_path, regression_path):
            pca_model = load_pca_model(pca_path)
            regression_model = load_model(regression_path)
            return pca_model, regression_model

        pca_model, regression_model = load_models(PCA_MODEL_PATH, REGRESSION_MODEL_PATH)
        if pca_model and regression_model:
            st.success("Model berhasil dimuat!")
            
            # Transformasi data dengan PCA
            st.write("Transformasi data dengan PCA...")
            X_test_pca = apply_pca_model(pca_model, X_test)
            
            # Prediksi dengan model regresi
            st.write("Melakukan prediksi...")
            Y_pred = regression_model.predict(X_test_pca)
            st.text_area('Penjelasan Model',"Model yang digunakan adalah Random Forest dengan alasan yang sudah dijabarkan pada bagian analisis data. kemudian digunakan juga multioutputregressor agar tujuan multi target tercapai secara otomatis. kemudian sebelum melakukan modeling, dilakukan PCA pada data train kemudian dibagi train_test_split untuk mengukur validasi model secara keseluruhan. pada proses sebelumnya sudah dilakukan juga Kfold dan CV pada model yang digunakan agar mendapatkan gambaran validasi yaitu sekitar di 3% (bisa dicek pada notebook 03_Modeling)")
            
            # Evaluasi model
            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(Y_test, Y_pred)
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            
            # Pilihan target feature
            selected_target = st.selectbox("Pilih Target Feature untuk Visualisasi:", target_columns)
            
            if selected_target:
                # Scatter plot untuk feature terpilih
                st.subheader(f"Perbandingan Y Asli vs Y Prediksi untuk {selected_target}")
                fig, ax = plt.subplots()
                ax.scatter(range(len(Y_test[selected_target])), Y_test[selected_target], alpha=0.6, label="Y Asli", color="blue")
                ax.scatter(range(len(Y_test[selected_target])), Y_pred[:, target_columns.index(selected_target)], alpha=0.6, label="Y Prediksi", color="orange")
                ax.set_xlabel("Index Data")
                ax.set_ylabel("Nilai")
                ax.set_title(f"Scatter Plot: {selected_target}")
                ax.legend()
                st.pyplot(fig)
