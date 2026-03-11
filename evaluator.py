import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

def evaluate_model(model, x_test, y_test, task):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():

        outputs=model(x_test)

        if task == "classification":
            _, predicted=torch.max(outputs, 1)
        else:
            predicted = outputs.squeeze()

    y_true=y_test.cpu().numpy()
    y_pred=predicted.cpu().numpy()

    if task == "classification":

        accuracy=accuracy_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return accuracy

    else:

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("MSE:", mse)
        print("R2 Score:", r2)
        
        return r2