from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

# Placeholder for x, y, T - you need to replace these with your actual data
# Example: If you have 100 samples with 5 features, a binary treatment, and a continuous outcome
x = np.random.rand(100, 5) 
T = np.random.randint(0, 2, 100) 
y = np.random.rand(100) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_res = np.zeros(len(y))
T_res = np.zeros(len(T))
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index],
    T_train, T_test = T[train_index], T[test_index],
    y_train, y_test = y[train_index], y[test_index],
model_y = RandomForestRegressor(n_estimators=100)
model_y.fit(np.column_stack((x_train, T_train)), y_train)
y_hat = model_y.predict(np.column_stack((x_test, T_test)))
model_T = RandomForestClassifier(n_estimators=100)
model_T.fit(x_train, T_train)
T_hat = model_T.predict_proba(x_test)[:, 1]
y_res[test_index] = y_test-y_hat
T_res[test_index] = T_test-T_hat
print("y_res:", y_res[test_index])
print("T_res:", T_res[test_index])
