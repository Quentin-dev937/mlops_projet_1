import pytest
from src.data.load_data import load_data
import pandas as pd

@pytest.fixture(scope="module")
def data_loader():

    X, y = load_data()

    return X, y

def test_load_data_returns_dataframe(data_loader):
    """Vérifie que load_data renvoie bien des DataFrames."""
    X, y = data_loader
    
    assert isinstance(X, pd.DataFrame), "X n'est pas un DataFrame"
    assert isinstance(y, pd.Series), "y n'est pas une Series"
    assert not X.empty, "Le DataFrame X est vide"
    print("✅ test_load_data_passed")

def test_load_data_columns(data_loader):
    """Vérifie que les données ont le bon nombre de colonnes (ex: 4 pour Iris)."""
    X, y = data_loader
    assert X.shape[1] == 4, f"Attendu 4 colonnes, obtenu {X.shape[1]}"