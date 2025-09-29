import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import load_data, preprocess_data, split_data
import numpy as np

df = load_data("data/diabetes_data.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1],1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=5, batch_size=64)

model.save("results/cnn_model.h5")
