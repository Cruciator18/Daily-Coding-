import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[3386,], [4127,], [5746,], [6391,], [7345,], [10026,], [12044,], [22016,], [29764,], [49377,]])
y = np.array([60.3, 62.4, 64.3, 65.8, 67.5, 70.1, 71.8, 76.4, 78.2, 81.3])



model = LinearRegression()

model.fit(X , y)


new_x = [[45322]]
prediction = model.predict(new_x)

print(f"Prediction : {prediction}")

plt.figure(figsize=(13,10))
plt.xlabel("Value of x")
plt.ylabel("Value of y (predictions)")
plt.scatter(X,y , color = 'blue' ,label = "Actual data")
plt.plot(X, model.predict(X) , color ='red' , linewidth = 2)
plt.legend(True)
plt.show()