import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генерация случайных данных
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказание
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# Визуализация
plt.scatter(X, y)
plt.plot(X_new, y_predict, color='red')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
