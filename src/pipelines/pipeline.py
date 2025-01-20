import os,sys
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data.load_data import load_arff_data
from src.models.load_pca_model import load_pca_model, apply_pca_model
from src.models.load_model import load_model

def main():
    # # Path ke file dan model
    TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "Supply Chain Management_test.arff")
    PCA_MODEL_PATH = os.path.join(ROOT_DIR, "models", "pca_model.pkl")
    REGRESSION_MODEL_PATH = os.path.join(ROOT_DIR, "models", "multi_target_model.pkl")

    # Load data test
    print("Memuat data test...")
    df = load_arff_data(TEST_DATA_PATH)
    if df is None:
        print("Gagal memuat data test. Menghentikan pipeline.")
        return

    # Pisahkan fitur dan target
    print("Memisahkan fitur dan target...")
    feature_columns = [col for col in df.columns if not (col.startswith("MTLp") or col == "LBL")]
    target_columns = [col for col in df.columns if col.startswith("MTLp") or col == "LBL"]
    
    X_test = df[feature_columns].astype(float)
    Y_test = df[target_columns].astype(float)

    # Load PCA model
    print("Memuat model PCA...")
    pca_model = load_pca_model(PCA_MODEL_PATH)
    if pca_model is None:
        print("Gagal memuat model PCA. Menghentikan pipeline.")
        return

    # Transformasi data test dengan PCA
    print("Transformasi data test dengan PCA...")
    X_test_pca = apply_pca_model(pca_model, X_test)
    if X_test_pca is None:
        print("Gagal menerapkan PCA pada data test. Menghentikan pipeline.")
        return

    # Load model
    print("Memuat model multitarget...")
    regression_model = load_model(REGRESSION_MODEL_PATH)
    if regression_model is None:
        print("Gagal memuat model regresi multitarget. Menghentikan pipeline.")
        return

    # Prediksi
    print("Melakukan prediksi pada data test")
    Y_pred = regression_model.predict(X_test_pca)

    # Evaluasi model dengan MAPE
    print("Menghitung MAPE...")
    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    print(f"Mean Absolute Percentage Error (MAPE) on Test Data: {mape:.2f}%")

if __name__ == "__main__":
    main()