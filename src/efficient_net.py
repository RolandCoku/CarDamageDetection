import tensorflow as tf
import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURATION ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 20


def run_efficient_net():
    print(f"Loading Data ({IMG_HEIGHT}x{IMG_WIDTH})...")
    X_train, y_train = preprocessing.load_data('../data/training')
    X_val, y_val = preprocessing.load_data('../data/validation')

    print("\nDownloading EfficientNetB0 (Smarter Brain)...")

    # EfficientNetB0 is a state-of-the-art model for image classification
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # We unfreeze the top 20 layers immediately for better adaptation
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Heavy Augmentation helps reach 95%
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2),

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),  # Helps stabilize training
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining High-Precision Model...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('efficientnet_model.keras', save_best_only=True)
    ]

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              validation_data=(X_val, y_val),
              callbacks=callbacks)

    # Evaluate
    predictions_prob = model.predict(X_val)
    predictions = (predictions_prob > 0.5).astype("int32")
    acc = accuracy_score(y_val, predictions)

    print(f"\n--- EfficientNet Accuracy: {acc * 100:.2f}% ---")


if __name__ == "__main__":
    run_efficient_net()