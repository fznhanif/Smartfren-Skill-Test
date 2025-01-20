import os
import pandas as pd
from scipy.io import arff

def load_arff_data(file_path):
    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
if __name__ == "__main__":
    file_path = 'data/Supply Chain Management_test.arff'
    print("Current working directory:", os.getcwd())
    df = load_arff_data(file_path)
    if df is not None:
        print("Data successfully loaded. First 5 rows:")
        print(df.head())
    else:
        print("Failed to load data.")
