import joblib

def load_model(model_path):
    """
    Memuat model regresi multitarget dari file .pkl.
    
    Args:
        model_path (str): Path ke file model .pkl.
    
    Returns:
        object: Model yang telah dimuat atau None jika gagal.
    """
    try:
        # Memuat model dari file .pkl
        model = joblib.load(model_path)
        print("Model regresi multitarget berhasil dimuat!")
        return model
    except Exception as e:
        print(f"Error saat memuat model dari {model_path}: {e}")
        return None

if __name__ == "__main__":
    # Contoh penggunaan
    model_path = "../../models/multi_target_model.pkl"
    regression_model = load_model(model_path)

    if regression_model:
        print("Model siap digunakan.")
    else:
        print("Gagal memuat model.")
