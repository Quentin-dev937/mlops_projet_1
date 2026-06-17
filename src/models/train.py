import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
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
    mlflow.set_experiment_tags({"version": "1.0",
                                "model": "rf"})

    print(f"🚀 Démarrage de l'expérience MLflow : {experiment_name}")
    print(f"📍 Tracking URI : {mlflow_uri}")

    print("2. Chargement des données...")
    X, y = load_data()

    print("3. Split des données...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    with mlflow.start_run() as run:

        run_id = run.info.run_id

        data_train = pd.concat([X_train, y_train], axis=1)
        dataset_train = mlflow.data.from_pandas(data_train, source="iris.csv")
        mlflow.log_input(dataset=dataset_train,
                         context="training")

        data_val = pd.concat([X_val, y_val], axis=1)
        dataset_val = mlflow.data.from_pandas(data_val, source="iris.csv")
        mlflow.log_input(dataset=dataset_val,
                         context="validation")
        
        print("4. Entraînement du modèle...")

        params = {"n_estimators": 10,
                  "random_state": 42,
                  "max_depth": 4}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("5. Évaluation...")
        y_pred = model.predict(X_val)
        accuracy, f1 = evaluate_model(y_val, y_pred)

        print("5. Logging du modèle")
        signature = infer_signature(X_val, y_pred)
        model_info = mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_val)
        
        mlflow.log_metrics({"accuracy": accuracy,
                            "f1-score": f1}, model_id=model_info.model_id, run_id=run_id, dataset=dataset_val)
        print(f"   📊 accuracy: {accuracy}")
        print(f"   📊 F1-score: {f1}")
        
        print(f"✅ Modèle mlflow sauvegardé lors du run {run_id}")

if __name__ == "__main__":
    train()