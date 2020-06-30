import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

X = np.array([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
y = np.array([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

X_scaled = preprocessing.scale(X).reshape(1,-1)

model = LinearRegression()
model.fit(X_scaled, y.reshape(1,-1))
predictions = model.predict(X_scaled)
print(y)
print(predictions)

style.use('ggplot')
plt.scatter(X, y, color='g', marker='o', s=15, label='Labels')
plt.plot(X, predictions.reshape(12), color='r', label='sklearn model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
