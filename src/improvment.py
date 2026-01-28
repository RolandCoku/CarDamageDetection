import tensorflow as tf
import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURATION ---
IMG_HEIGHT = 64
IMG_WIDTH = 64
EPOCHS = 50


def run_improvement():
    print("Loading Data...")
    X_train, y_train = preprocessing.load_data('../data/training')
    X_val, y_val = preprocessing.load_data('../data/validation')

    print("\nTraining Improved CNN (with Callbacks)...")

    # Define the Model (Same as before)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- THE FIX: CALLBACKS ---
    callbacks = [
        # Stop training if val_loss doesn't improve for 10 epochs
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
        # Save the best version automatically
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]

    # Train with callbacks
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks  # Add them here
    )

    # The model variable now holds the "best" weights (restored by EarlyStopping)
    predictions_prob = model.predict(X_val)
    predictions = (predictions_prob > 0.5).astype("int32")

    acc = accuracy_score(y_val, predictions)
    print(f"\n--- Best Model Accuracy: {acc * 100:.2f}% ---")

    cm = confusion_matrix(y_val, predictions)
    print(cm)


if __name__ == "__main__":
    run_improvement()