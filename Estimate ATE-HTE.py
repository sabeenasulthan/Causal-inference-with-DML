import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
np.random.seed(42)
n = 2000 
p = 20 
x = np.random.normal(0, 1, size=(n, p))
true_tau = 2+x[:, 0]-0.5*x[:, 1] 
logits = 0.5*x[:, 0]-0.25*x[:, 2]
propensity = 1/(1+np.exp(-logits))
T = np.random.binomial(1, propensity)
y0 = x[:, 0]+x[:, 1]+np.random.normal(0, 1, n)
y = y0+T*true_tau
data = pd.DataFrame(x, columns=[f"x{i}" for i in range(p)])
data['T'] = T
data['y'] = y
data['true_tau'] = true_tau
data.head()
y = data['y']
T = data['T']
X = data[['x1', 'x2']] # Assuming 'x1' and 'x2' are the covariates
# First stage: Regress T on X to get residuals T_res
model_T = LinearRegression()
model_T.fit(X, T)
T_pred = model_T.predict(X)
T_res = T - T_pred
# First stage: Regress y on X to get residuals y_res
model_y = LinearRegression()
model_y.fit(X, y)
y_pred = model_y.predict(X)
y_res = y - y_pred
# Task3"Estimate ATE & HTE
# ATE(final stage Regression)
final_model = LinearRegression()
final_model.fit(T_res.values.reshape(-1, 1), y_res)
ATE_DML = final_model.coef_[0]
print("ATE DML:", ATE_DML)

# HTE (Subgroup based)
# Make a copy to avoid SettingWithCopyWarning if 'data' is a slice
data_copy = data.copy()
data_copy['group'] = data_copy['x2'] > data_copy ['x2'].median()
HTE_low = data_copy[data_copy['group'] == 0]['y'].mean()
HTE_high = data_copy[data_copy['group'] == 1]['y'].mean()
print("HTE (low x2):", HTE_low)
print("HTE (highx2):", HTE_high)
# Task4: compare with ols
x_ols = sm.add_constant(data[['T', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5']])
x = sm.add_constant(x)
ols_model = sm.OLS(data['y'], x_ols).fit()
print("OLS Treatment Effect:", ols_model.params['T'])

# CREATE Summary Table
summary = pd.DataFrame({
    "method": ["DML", "OLS"],
    "ATE_Estimate": [ATE_DML, ols_model.params["T"]],
     "Bias_vs_True": [abs(ATE_DML - true_tau.mean()), abs(ols_model.params["T"] - true_tau.mean())]})
print(summary)

