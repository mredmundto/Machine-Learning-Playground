import pandas as pd
cars = pd.read_csv("auto.csv")
print(cars.head())
unique_regions = cars["origin"].unique()
print(unique_regions)

#creating new columns based on cylinder column 
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)

#same here for year 
dummy_years = pd.get_dummies(cars["year"], prefix="year")
cars = pd.concat([cars, dummy_years], axis=1)
cars = cars.drop("year", axis=1)
cars = cars.drop("cylinders", axis=1)
print(cars.head())


import numpy as np 
#randomizing the sample and put 70% and 30% as training and test 
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]


#training with binary classifier north america = 1 and the rest = 0 
from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()

models = {}
features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]

for origin in unique_origins:
    model = LogisticRegression()
    
    X_train = train[features]
    y_train = train["origin"] == origin

    model.fit(X_train, y_train)
    models[origin] = model

testing_probs = pd.DataFrame(columns=unique_origins)
testing_probs = pd.DataFrame(columns=unique_origins)  

for origin in unique_origins:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]


#Now that we trained the models and computed the probabilities in each origin we can classify each observation. To classify each observation we want to select the origin with the highest probability of classification for that observation.
# While each column in our dataframe testing_probs represents an origin we just need to choose the one with the largest probability. We can use the Dataframe method .idxmax() to return a Series where each value corresponds to the column or where the maximum value occurs for that observation. We need to make sure to set the axis paramater to 1 since we want to calculate the maximum value across columns. Since each column maps directly to an origin the resulting Series will be the classification from our model.

# Classify each observation in the test set using the testing_probs Dataframe.
predicted_origins = testing_probs.idxmax(axis=1)
# Assign the predicted origins to predicted_origins and use the print function to display it.
print(predicted_origins)
