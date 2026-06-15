import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

# Import des modules internes
from src.data.load_data import load_data
from src.evaluation.evaluate import evaluate_model


def train():
    print("1. Configuration de mlflow...")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = "mlops_project_1"
    mlflow.set_experiment(experiment_name)

    print(f"🚀 Démarrage de l'expérience MLflow : {experiment_name}")
    print(f"📍 Tracking URI : {mlflow_uri}")

    with mlflow.start_run():
        print("2. Chargement des données...")

        X, y = load_data()
        
        print("3. Split des données...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("4. Entraînement du modèle...")

        params = {"n_estimators": 10,
                  "random_state": 42,
                  "max_depth": 4}
                
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print("5. Évaluation...")
        y_pred = model.predict(X_test)
        accuracy = evaluate_model(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        print(f"   📊 accuracy: {accuracy}")
        
        print("6. Sauvegarde du modèle...")
        signature = infer_signature(X_test, y_pred)
        model_info = mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X)
        print(f"✅ Modèle mlflow sauvegardé lors du run {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()