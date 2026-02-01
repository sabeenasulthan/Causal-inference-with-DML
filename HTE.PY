import pandas as pd
import numpy as np
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
print(data.head())







