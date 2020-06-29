import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.figure import Figure

def standardize_data(data):
    return (data - np.mean(data)) / np.std(data)

class LinearRegression:

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        
    def fit(self, X_train, y_train, learning_rate, epochs):
        self.frac = 1/(2*len(y_train))
        for i in range(epochs):
            self.hyp_vec = X_train*self.theta1 + self.theta0
            self.J = sum((self.hyp_vec - y_train)**2)
            print('===== Cost : {} ====== Epoch : {} ====='.format(self.J, i+1))
            self.d0 = self.frac * learning_rate * sum(self.hyp_vec - y_train)
            self.d1 = self.frac * learning_rate * np.dot((self.hyp_vec - y_train), X_train)
            self.theta0 -= self.d0
            self.theta1 -= self.d1
                        
    def predict(self, data):
        self.predictions = self.theta1 * data + self.theta0
        return self.predictions

    def error(self, y_test, predictions):
        return sum((y_test-predictions)**2)


X = np.array([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
y = np.array([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

formula_m = (np.mean(X)*np.mean(y) - np.mean(X*y)) / (np.mean(X)**2 - np.mean(X**2))
formula_b = np.mean(y) - formula_m*np.mean(X)
formula_predictions = X*formula_m + formula_b

model = LinearRegression()
model.fit(standardize_data(X), y, learning_rate=0.05, epochs=2000)
model_predictions = model.predict(standardize_data(X))

print(sum((model_predictions - y)**2))
print(sum((formula_predictions - y)**2))

style.use('ggplot')

fig = plt.figure()

ax1 = fig.add_subplot(1.5, 2, 1)
ax1.scatter(X, y, s=15, marker='o', color='g', label='Labels')
ax1.plot(X, model_predictions, label='Predictions')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression')

ax2 = fig.add_subplot(1.5, 2, 2)
ax2.scatter(X, y, s=15, marker='o', color='g', label='Labels')
ax2.scatter(X, formula_predictions, s=15, marker='o', color='r', label='Predictions')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Formula Based Model')

plt.legend()
plt.show()






