from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

def select_model(x_train, y_train, x_test, y_test, task='classification'):

    results={}

    if task == 'classification':
        # Logistic Regression
        lr=LogisticRegression(max_iter=1000)
        lr.fit(x_train, y_train)
        lr_pred=lr.predict(x_test)
        results['Logistic Regression']=accuracy_score(y_test, lr_pred)

        # Random Forest
        rf=RandomForestClassifier(n_estimators=100)
        rf.fit(x_train, y_train)
        rf_pred=rf.predict(x_test)
        results['Random Forest']=accuracy_score(y_test, rf_pred)
        
        # Gradient Boosting
        gb=GradientBoostingClassifier(n_estimators=100)
        gb.fit(x_train, y_train)
        gb_pred=gb.predict(x_test)
        results['Gradient Boosting']=accuracy_score(y_test, gb_pred)
        
        # SVM
        svm=SVC(kernel='rbf')
        svm.fit(x_train, y_train)
        svm_pred=svm.predict(x_test)
        results['SVM']=accuracy_score(y_test, svm_pred)
    
    else:  # regression
        # Linear Regression
        lr=LinearRegression()
        lr.fit(x_train, y_train)
        lr_pred=lr.predict(x_test)
        results['Linear Regression']=r2_score(y_test, lr_pred)
        
        # Ridge Regression
        ridge=Ridge(alpha=1.0)
        ridge.fit(x_train, y_train)
        ridge_pred=ridge.predict(x_test)
        results['Ridge']=r2_score(y_test, ridge_pred)

        # Random Forest
        rf=RandomForestRegressor(n_estimators=100)
        rf.fit(x_train, y_train)
        rf_pred=rf.predict(x_test)
        results['Random Forest']=r2_score(y_test, rf_pred)
        
        # Gradient Boosting
        gb=GradientBoostingRegressor(n_estimators=100)
        gb.fit(x_train, y_train)
        gb_pred=gb.predict(x_test)
        results['Gradient Boosting']=r2_score(y_test, gb_pred)
        
        # SVR
        svr=SVR(kernel='rbf')
        svr.fit(x_train, y_train)
        svr_pred=svr.predict(x_test)
        results['SVR']=r2_score(y_test, svr_pred)

    best_model=max(results, key=results.get)
    metric_name = "Accuracy" if task == "classification" else "R² Score"
    print(f"Best Model: {best_model} with {metric_name}: {results[best_model]:.4f}")
    return results, best_model