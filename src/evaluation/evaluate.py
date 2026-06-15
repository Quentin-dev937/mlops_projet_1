from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(y_true, y_pred):
    """Calcule et affiche les métriques."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print(classification_report(y_true, y_pred))
    return acc