import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():

        outputs=model(x_test)
        _, predicted=torch.max(outputs, 1)

    y_true=y_test.cpu().numpy()
    y_pred=predicted.cpu().numpy()

    accuracy=accuracy_score(y_true, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nconfusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification_report:")
    print(classification_report(y_true, y_pred, zero_division=0))