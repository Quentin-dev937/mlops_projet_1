import pytest
import numpy as np
from src.evaluation.evaluate import evaluate_model
from src.models.train import train
import os

# --- FIXTURE (Préparation des données de test) ---
@pytest.fixture(scope="module")
def sample_predictions():
    """
    Génère des données de test fictives (y_true, y_pred) pour valider la fonction evaluate.
    On n'a pas besoin d'entraîner un vrai modèle pour tester la fonction de calcul de métriques.
    C'est plus rapide et plus isolé.
    """
    # Données fictives : 5 échantillons
    # Vérités terrain : [0, 1, 1, 0, 2]
    y_true = np.array([0, 1, 1, 0, 2])
    
    # Prédictions parfaites (pour tester le cas idéal)
    y_pred_perfect = np.array([0, 1, 1, 0, 2])
    
    # Prédictions imparfaites (pour tester le calcul réel)
    y_pred_imperfect = np.array([0, 1, 0, 0, 2]) # Une erreur sur le 3ème élément
    
    return y_true, y_pred_perfect, y_pred_imperfect

# --- TESTS ---

def test_evaluate_returns_float(sample_predictions):
    """
    Test 1 : Vérifie que la fonction renvoie bien un nombre flottant (float).
    """
    y_true, _, y_pred_imperfect = sample_predictions
    
    accuracy = evaluate_model(y_true, y_pred_imperfect)
    
    assert isinstance(accuracy, float), f"L'accuracy devrait être un float, pas {type(accuracy)}"

def test_evaluate_accuracy_range(sample_predictions):
    """
    Test 2 : Vérifie que l'accuracy est bien comprise entre 0 et 1.
    C'est une contrainte mathématique fondamentale.
    """
    y_true, _, y_pred_imperfect = sample_predictions
    
    accuracy = evaluate_model(y_true, y_pred_imperfect)
    
    assert 0.0 <= accuracy <= 1.0, f"L'accuracy {accuracy} n'est pas entre 0 et 1"

def test_evaluate_perfect_score(sample_predictions):
    """
    Test 3 : Vérifie que si les prédictions sont parfaites, l'accuracy est de 1.0.
    """
    y_true, y_pred_perfect, _ = sample_predictions
    
    accuracy = evaluate_model(y_true, y_pred_perfect)
    
    assert accuracy == 1.0, f"Une prédiction parfaite devrait donner 1.0, obtenu {accuracy}"

def test_evaluate_handles_shape_mismatch():
    """
    Test 4 : Vérifie que la fonction gère (ou lève une erreur claire) si les tailles ne correspondent pas.
    C'est un test de robustesse crucial en production.
    """
    y_true = np.array([0, 1, 1])
    y_pred_wrong_size = np.array([0, 1]) # Taille différente
    
    # On s'attend à ce que scikit-learn lève une ValueError
    with pytest.raises(ValueError):
        evaluate_model(y_true, y_pred_wrong_size)