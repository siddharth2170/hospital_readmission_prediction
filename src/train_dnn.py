import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import load_data, preprocess_data, split_data

df = load_data("data/diabetes_data.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=5, batch_size=64)

model.save("results/dnn_model.h5")
