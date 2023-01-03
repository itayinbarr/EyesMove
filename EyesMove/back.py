import keras.models
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import tensorflow as tf


# Build model and train it
def create_model():
    # Load data into training and eval
    data = pd.read_csv('../data/train.csv')

    # Extract the electrodes and labels
    electrodes = data.iloc[:, :-1]
    labels = data.iloc[:, -1]  # all rows, only the last column

    # Normalize electrodes
    electrodes = (electrodes - electrodes.mean()) / electrodes.std()

    # Split the data into a training set and a test set
    train_electrodes, test_electrodes, train_labels, test_labels = train_test_split(electrodes, labels, test_size=0.2)

    # Build the model
    model = tf.keras.Sequential()
    # Input layer, back propagation is relu
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(electrodes.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # Output layer, back propagation is sigmoid
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_electrodes, train_labels, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_electrodes, test_labels, batch_size=32)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Saving model
    model.save('eye_movement')


def use_model(patient):
    model = keras.models.load_model('eye_movement')
    data = pd.read_csv('../data/test.csv')
    # Normalize electrodes
    data = (data - data.mean()) / data.std()
    result = model.predict(data)
    pred = result[patient]*100
    if pred > 50:
        print('the eyes are open with ' + str(pred) + ' percent')
    else:
        print('the eyes are closed with ' + str(100-pred) + ' percent')
