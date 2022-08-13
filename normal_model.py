import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

red_wine = pd.read_csv("winequality-red.csv")

ct = MinMaxScaler()
#Create X and y
X = red_wine.drop("quality", axis = 1)
y = red_wine["quality"]

#Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train = ct.fit(X_train)

X_train_normal = ct.fit_transform(X_train)
X_test_normal = ct.fit_transform(X_test)

#Create a model
wine_quality = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="sigmoid"),
    tf.keras.layers.Dense(20, activation="sigmoid"),
    tf.keras.layers.Dense(10, activation="sigmoid"),
    tf.keras.layers.Dense(1)
])

wine_quality.compile(loss = 'mae',
                     optimizer =tf.keras.optimizers.Adam(lr = 0.01),
                     metrics = ["mae"])


wine_quality.fit(X_train_normal, y_train, epochs = 200)

wine_quality.evaluate(X_test_normal, y_test)

wine_quality.save("normal_model.h5")