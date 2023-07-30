from xgboost import XGBClassifier, XGBRegressor

def xgboost_classifier(X_train, y_train, X_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def xgboost_regressor(X_train, y_train, X_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred