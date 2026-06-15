import pandas as pd
import os

DATA_PATH = os.path.join("data", "raw", "iris.csv")

def load_data():
    """Charge les données brutes et retourne X et y."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier non trouvé : {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Séparation features / target (à adapter selon vos colonnes)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    
    return X, y