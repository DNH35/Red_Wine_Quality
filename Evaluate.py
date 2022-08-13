import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
import numpy as np

red_wine = pd.read_csv("winequality-red.csv")
ct = MinMaxScaler()
#Create X and y
X = red_wine.drop("quality", axis = 1)
y = red_wine["quality"]

#Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_normal = ct.fit_transform(X_train)
X_test_normal = ct.fit_transform(X_test)

first_model = tf.keras.models.load_model("model_1.h5")
first_model.evaluate(X_test, y_test)

second_model = tf.keras.models.load_model("model_2.h5")
second_model.evaluate(X_test, y_test)

third_model = tf.keras.models.load_model("model_3.h5")
third_model.evaluate(X_test, y_test)

normal_model = tf.keras.models.load_model("normal_model.h5")
normal_model.evaluate(X_test_normal, y_test)

y_pred = np.array(normal_model.predict(X_test_normal), dtype=np.int32)