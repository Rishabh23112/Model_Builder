from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def select_model(x_train, y_train, x_test, y_test):

    results={}

    lr=LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)

    lr_pred=lr.predict(x_test)
    results['Logistic Regression']=accuracy_score(y_test, lr_pred)


    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(x_train, y_train)

    rf_pred=rf.predict(x_test)
    results['Random Forest']=accuracy_score(y_test, rf_pred)

    best_model=max(results, key=results.get)
    print(f"Best Model: {best_model} with Accuracy: {results[best_model]:.4f}")
    return results, best_model