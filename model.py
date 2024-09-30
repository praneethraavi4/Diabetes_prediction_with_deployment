import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv")
print(df)
print(df.columns)

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]

Y = df[['Outcome']]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2)

model = Sequential([
    Input(shape=(8,)),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 16, activation = 'relu'),
    Dense(units = 1, activation = 'sigmoid')

])

model.compile(optimizer = 'Adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
history = model.fit(X_train,Y_train,epochs  = 10, validation_split = 0.1, verbose = 2)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(10), accuracy, label='Training Accuracy')
plt.plot(range(10), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(10), loss, label='Training Loss')
plt.plot(range(10), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('model.h5')