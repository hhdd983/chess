import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_chess_nn_model():
    model = models.Sequential([
        layers.Input(shape=(64 * 12,)),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64 * 64, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model(X_train, y_train, model_path='chess_ai_model.h5'):
    print("Training model...")
    model = create_chess_nn_model()
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save(model_path)
    print(f"Model trained and saved as {model_path}")
    return model

if __name__ == "__main__":
    print("Loading training data...")
    try:
        X_data = np.load('X_train_lichess_api.npy')
        y_data = np.load('y_train_lichess_api.npy')
        print(f"Data loaded. X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")
        
        # 모델 학습 및 저장
        train_and_save_model(X_data, y_data)

    except FileNotFoundError:
        print("Error: Training data (X_train_lichess_api.npy, y_train_lichess_api.npy) not found.")
        print("Please run data_collector.py first to generate and save the data.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")