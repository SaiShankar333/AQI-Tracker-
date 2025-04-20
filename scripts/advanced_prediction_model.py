from sklearn.ensemble import GradientBoostingRegressor

def train_gradient_boost(X, y):
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model
