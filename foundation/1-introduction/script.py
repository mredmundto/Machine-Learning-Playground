import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
print(cars.head(5))

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
cars.plot("weight", "mpg", kind='scatter', ax=ax1)
cars.plot("acceleration", "mpg", kind='scatter', ax=ax2)
# plt.show()

import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(cars[["weight"]], cars["mpg"])
predictions = lr.predict(cars[["weight"]])

print("printing the values of prediction and actual")
print(predictions[0:5])
print(cars["mpg"][0:5])

plt.scatter(cars["weight"], cars["mpg"], c='red')
plt.scatter(cars["weight"], predictions, c='blue')
plt.xlim(1000, 5500)
# plt.show()


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(cars["mpg"], predictions)
print("mean squared error")
print(mse)


mse = mean_squared_error(cars["mpg"], predictions)
rmse = mse ** (0.5)
print("Root mean squared error")
print(rmse)
