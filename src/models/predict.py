import os
import skops.io as sio
import mlflow

# Chemin relatif vers le modèle entraîné
#MODEL_PATH = os.path.join("models", "model.skops")
#model_info.model_uri = "models/mlflow_model"

def predict_single_sample(input_data=None):
    """
    Charge le modèle le plus récent et prédit sur un échantillon.
    Si aucun modèle n'est trouvé, lance l'entraînement automatiquement.
    
    input_data : liste de valeurs (ex: [5.1, 3.5, 1.4, 0.2])
    """
    
    model = None
    model_uri = None
    
    # 1. Tenter de récupérer le modèle le plus récent existant
    experiment = mlflow.get_experiment_by_name("mlops_project_1")
    
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            print(f"🔍 Run existant trouvé : {run_id}")
            
            try:
                print(f"⏳ Chargement du modèle depuis : {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                print("✅ Modèle chargé avec succès.")

            except Exception as e:
                print(f"⚠️  Erreur inattendue : {e}. Relance de l'entraînement...")
    
    # 2. Si aucun modèle n'a pu être chargé, lancer l'entraînement
    if model is None:
        print("🚀 Aucun modèle valide trouvé. Lancement de l'entraînement automatique...")
        from src.models.train import train
        train()
        
        # Récupérer le NOUVEAU run qui vient d'être créé
        experiment = mlflow.get_experiment_by_name("mlops_project_1")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise RuntimeError("Échec critique : Aucun modèle trouvé même après l'entraînement.")
            
        new_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{new_run_id}/model"
        print(f"🔄 Chargement du nouveau modèle depuis : {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

    # 3. Données par défaut si rien n'est fourni
    if input_data is None:
        # Valeurs exemple (Iris)
        input_data = [[5.1, 3.5, 1.4, 0.2]] 
    elif isinstance(input_data, list) and not isinstance(input_data[0], list):
        # S'assurer que l'entrée est bien une liste de listes (batch)
        input_data = [input_data]
    
    # 4. Prédire
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    
    print(f"🔮 Prédiction : Classe {prediction[0]}")
    print(f"📊 Probabilités : {proba[0]}")
    
    return prediction[0]

if __name__ == "__main__":
    predict_single_sample()