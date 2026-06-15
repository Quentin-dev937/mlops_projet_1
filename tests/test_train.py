import os
import pytest
import numpy as np
from sklearn.base import BaseEstimator
from src.models.train import train
import mlflow

@pytest.fixture(scope="module")
def trained_model():
    #if os.path.exists("models"):
    #    os.remove("models")
    train()
    
    # 2. Récupérer le dernier run de l'expérience "mlops_project_1"
    experiment = mlflow.get_experiment_by_name("mlops_project_1")
    if not experiment:
        raise RuntimeError("L'expérience 'mlops_project_1' n'a pas été trouvée. Vérifiez le nom dans train.py.")
    
    # On cherche le run le plus récent (trié par start_time DESC)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise RuntimeError("Aucun run trouvé après l'exécution de train().")
    
    run_id = runs.iloc[0]["run_id"]
    
    # 3. Construire l'URI correcte
    # Le nom 'model' doit correspondre exactement à l'argument name="model" dans train.py
    model_uri = f"runs:/{run_id}/model"
    
    
    print(f"🔄 Chargement du modèle depuis l'URI : {model_uri}")
    
    # 4. Charger et retourner le modèle
    return mlflow.sklearn.load_model(model_uri)

def test_train_creates_model_file(trained_model):
    """Test 1 : Le fichier est-il créé ?"""

    assert trained_model is not None, "Le modèle n'a pas pu être chargé depuis MLflow."

def test_train_model_is_loadable(trained_model):
    """Test 2 : Le fichier est-il un vrai modèle ?"""

    model = trained_model
    assert isinstance(model, BaseEstimator), "Objet invalide."
    assert hasattr(model, "classes_"), "Modèle non entraîné."

def test_train_model_predicts_correctly(trained_model):
    """Test 3 : Le modèle fonctionne-t-il ?"""

    model = trained_model
    dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(dummy_input)
    assert len(prediction) == 1, "Erreur de format de prédiction."