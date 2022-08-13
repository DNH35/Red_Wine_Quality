import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

red_wine = pd.read_csv("winequality-red.csv")

#Create X and y
X = red_wine.drop("quality", axis = 1)
y = red_wine["quality"]

#Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a model
wine_quality = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(1)
])

wine_quality.compile(loss = 'mae',
                     optimizer = 'SGD',
                     metrics = ["mae"])

wine_quality.fit(X_train, y_train, epochs = 100)

wine_quality.save("model_2.h5")