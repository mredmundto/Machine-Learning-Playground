import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
filtered_cars = cars[cars['horsepower'] != '?']
filtered_cars['horsepower'] = filtered_cars['horsepower'].astype('float')


import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(filtered_cars[["horsepower"]], filtered_cars["mpg"])
predictions = lr.predict(filtered_cars[["horsepower"]])
print(predictions[0:5])
print(filtered_cars["mpg"][0:5].values)


import matplotlib.pyplot as plt
plt.scatter(filtered_cars["horsepower"], filtered_cars["mpg"], c='red')
plt.scatter(filtered_cars["horsepower"], predictions, c='blue')
plt.show()


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(filtered_cars["mpg"], predictions)
print(mse)
rmse = mse ** 0.5
print(rmse)