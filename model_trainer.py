# ---------- importing neccesary libraries ----------
import numpy as np
import tensorflow as tf
from keras import layers, models


# ---------- Creating A simple
def create_chess_nn_model():
    model = models.Sequential([  # Creating CNN model
        layers.Input(shape=(8, 8, 12)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),   # Fully connected layer
        # output layer (one hot encoding)
        layers.Dense(4096, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      label_smoothing=0.1),
                  metrics=['accuracy'])
    return model


def train_and_save_model(X_train, y_train, model_path='chess_ai_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model.")
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(
                          label_smoothing=0.1),
                      metrics=['accuracy'])
    except:
        print("No existing model found. Creating new one.")
        model = create_chess_nn_model()

    model.summary()

    # Checkpoint setup
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(
        "chess_ai_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint]
    )

    print("Training done.")
    return model


if __name__ == "__main__":  # only run it if this is directly ran
    print("Loading training data...")
    try:
        X_data = np.load('X_train_lichess_api.npy')
        y_data = np.load('y_train_lichess_api.npy')

        # reshaping x data from (N, 768) to (N, 8, 8, 12)
        X_data = X_data.reshape(-1, 8, 8, 12)
        print(
            f"Data loaded. X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")

        # saving the trained information
        train_and_save_model(X_data, y_data)

    except FileNotFoundError:
        print("Error: Training data (X_train_lichess_api.npy, y_train_lichess_api.npy) not found.")
        print("Please run data_collector.py first to generate and save the data.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
