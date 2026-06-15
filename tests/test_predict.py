import os
import pytest
from src.models.predict import predict_single_sample
from src.models.train import train

@pytest.fixture(scope="module")
def model_ready():
    """S'assure qu'un modèle est disponible pour les tests de prédiction."""
    #if not os.path.exists("models"):
    train()
    return True

def test_predict_returns_class(model_ready):
    # On n'appelle pas train() ici, on fait confiance à la fixture
    result = predict_single_sample([[5.1, 3.5, 1.4, 0.2]])
    assert result in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    #assert result in [0, 1, 2]