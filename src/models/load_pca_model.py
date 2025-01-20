import joblib
import pandas as pd

def load_pca_model(model_path):
    try:
        # Load the PCA model
        pca_model = joblib.load(model_path)
        print("PCA model loaded successfully!")
        return pca_model
    except Exception as e:
        print(f"Error loading PCA model: {e}")
        return None

def apply_pca_model(pca_model, data):
    try:
        transformed_data = pca_model.transform(data)
        return transformed_data
    except Exception as e:
        print(f"Error applying PCA model: {e}")
        return None

if __name__ == "__main__":
    model_path = '../../models/pca_model.pkl'
    pca_model = load_pca_model(model_path)

    if pca_model:
        data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "feature3": [7.0, 8.0, 9.0]
        })

        transformed_data = apply_pca_model(pca_model, data)
        print("Transformed Data:")
        print(transformed_data)
