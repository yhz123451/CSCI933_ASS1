import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("bike_rental_data.csv")

# Features and target
X = df.drop(columns=["bikes_rented"])
y = df["bikes_rented"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Statistical Learning ---
# Define the eval. model
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.2f}, R^2={r2:.2f}")
    return rmse, r2


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)


# Ridge Regression
print("\nRidge:")
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"The best alpha: {ridge_cv.alpha_}")
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train_scaled, y_train)


# Lasso Regression
print("\nLasso:")
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], cv=5)
lasso_cv.fit(X_train_scaled, y_train)
print(f"The best alpha: {lasso_cv.alpha_}")
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train_scaled, y_train)
print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")

# Elastic Net
print("\nElastic Net:")
enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], alphas=[0.001, 0.01, 0.1], cv=5)
enet_cv.fit(X_train_scaled, y_train)
print(f"The best l1_ratio: {enet_cv.l1_ratio_}, The best alpha: {enet_cv.alpha_}")
enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_)
enet.fit(X_train_scaled, y_train)

# Evaluate models
print("\nEvaluation:")
lin_rmse, lin_r2 = evaluate_model(lin_reg, X_test_scaled, y_test, "OLS Regression")
ridge_rmse, ridge_r2 = evaluate_model(ridge, X_test_scaled, y_test, "Ridge Regression")
lasso_rmse, lasso_r2 = evaluate_model(lasso, X_test_scaled, y_test, "Lasso Regression")
enet_rmse, enet_r2 = evaluate_model(enet, X_test_scaled, y_test, "Elastic Net")

# Analysis of statistical methods
# Comparison of feature
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'OLS': lin_reg.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
    'ElasticNet': enet.coef_
})
print("\nCOEF:")
print(coef_df.sort_values('OLS', ascending=False))

# --- Deep Learning Approach ---w
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


class LinearNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1) # Single-layer linear model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


def train_model(X_train, y_train, X_test, y_test,dropout_rate=0.0, weight_decay=0.0,epochs=100, lr=0.01):
    model = LinearNN(X_train.shape[1], dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        rmse = mean_squared_error(y_test.numpy(), y_pred, squared=False)
        r2 = r2_score(y_test.numpy(), y_pred)
    return rmse, r2, train_losses


# linear neural network
print("\nLinear neural network:")
nn_rmse, nn_r2, _ = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
print(f"NN Baseline: RMSE={nn_rmse:.2f}, R²={nn_r2:.2f}")

# weight decay
print("\nWeight decay :")
weight_decays = [0.001, 0.01, 0.1]
decay_rmses = []
for decay in weight_decays:
    rmse, r2, _ = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, weight_decay=decay)
    decay_rmses.append(rmse)
    print(f"WD={decay}: RMSE={rmse:.2f}, R²={r2:.2f}")
best_decay = weight_decays[decay_rmses.index(min(decay_rmses))]
nn_l2_rmse, nn_l2_r2, _ = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,weight_decay=best_decay)

# dropout
print("\nDropout:")
dropout_rates = [0.1, 0.3, 0.5]
dropout_rmses = []
for rate in dropout_rates:
    rmse, r2, _ = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, dropout_rate=rate)
    dropout_rmses.append(rmse)
    print(f"Dropout p={rate}: RMSE={rmse:.2f}, R²={r2:.2f}")
best_dropout = dropout_rates[dropout_rmses.index(min(dropout_rmses))]
nn_dropout_rmse, nn_dropout_r2, _ = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, dropout_rate=best_dropout)

# Feature Engineering
print("\nFeature Engineering:")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Standard new features
poly_scaler = StandardScaler()
X_poly_train_scaled = poly_scaler.fit_transform(X_poly_train)
X_poly_test_scaled = poly_scaler.transform(X_poly_test)
X_poly_train_tensor = torch.tensor(X_poly_train_scaled, dtype=torch.float32)
X_poly_test_tensor = torch.tensor(X_poly_test_scaled, dtype=torch.float32)

# Train new model
poly_rmse, poly_r2, _ = train_model(X_poly_train_tensor, y_train_tensor, X_poly_test_tensor, y_test_tensor)
print(f"With Polynomial Features: RMSE={poly_rmse:.2f}, R²={poly_r2:.2f}")

# Analysis
print("\nAnalysis")

results = pd.DataFrame({
    'Method': ['OLS', 'Ridge', 'Lasso', 'ElasticNet',
              'NN Baseline', 'NN (L2)', 'NN (Dropout)', 'NN (Poly)'],
    'RMSE': [lin_rmse, ridge_rmse, lasso_rmse, enet_rmse,
             nn_rmse, nn_l2_rmse, nn_dropout_rmse, poly_rmse],
    'R²': [lin_r2, ridge_r2, lasso_r2, enet_r2,
           nn_r2, nn_l2_r2, nn_dropout_r2, poly_r2]
})

print("\nDeep Learning vs. Statistical Approaches:")
print(results.sort_values('RMSE'))


import matplotlib.pyplot as plt

def plot_predictions(models, model_names, X_test_scaled, y_test):
    plt.figure(figsize=(10, 6))
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test_scaled)
        plt.scatter(y_test, y_pred, label=name, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted Values (Statistical Models)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.show()


def plot_residuals(model, X_test_scaled, y_test, model_name):
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(f'residuals_{model_name.lower()}.png')
    plt.show()


def plot_training_curve(train_losses, title="NN Training Loss Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def plot_rmse_bar(results_df):
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Method'], results_df['RMSE'], color='skyblue')
    plt.xlabel('RMSE')
    plt.title('Model RMSE Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

plot_predictions(
    [lin_reg, ridge, lasso, enet],
    ["OLS", "Ridge", "Lasso", "ElasticNet"],
    X_test_scaled, y_test
)

for m, name in zip([lin_reg, ridge, lasso, enet], ["OLS", "Ridge", "Lasso", "ElasticNet"]):
    plot_residuals(m, X_test_scaled, y_test, name)

_, _, losses_baseline = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
plot_training_curve(losses_baseline, "NN Baseline Training Loss")

plot_rmse_bar(results)


