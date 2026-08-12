import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

predicted = model.predict([[6]])
print("Predicted values:", predicted[0])

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')

out_path = 'exam_score_plot.png'
plt.savefig(out_path)
print('Saved plot:', out_path)
plt.close()
