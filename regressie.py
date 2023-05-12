import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('regressie_tabel.csv')

x = df['QUANTITY'].values.reshape(-1, 1)
y = df['RETURN_QUANTITY'].values

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)

print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

plt.scatter(x, y, color='b', label='Data Points')
plt.plot(x, model.predict(x), color='r', label='Linear Regression')
plt.xlabel('QUANTITY')
plt.ylabel('RETURN_QUANTITY')
plt.title('Linear Regression of RETURN_QUANTITY vs QUANTITY')
plt.legend()
plt.show()
