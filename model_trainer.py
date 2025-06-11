import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def create_chess_nn_model():
    model = models.Sequential([
        layers.Input(shape=(8, 8, 12)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def train_and_save_model(X_train, y_train, model_path='chess_ai_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model.")
    except:
        print("No existing model found. Creating new one.")
        model = create_chess_nn_model()

    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    lr_schedule = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train,
        y_train,
        epochs = 50,
        batch_size = 64,
        validation_split = 0.2,
        shuffle = True,
        callbacks=[checkpoint, lr_schedule, early_stop]
    )

    print("Training done.")
    return model


if __name__ == "__main__":
    print("Loading training data...")
    try:
        X_data = np.load('training_data/X_players.npy')
        y_data = np.load('training_data/y_players.npy')

        X_data = X_data[:100000]
        y_data = y_data[:100000]

        X_data = X_data.reshape(-1, 8, 8, 12)
        print(
            f"Data loaded. X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")

        train_and_save_model(X_data, y_data)

    except FileNotFoundError:
        print("Error: Training data not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
