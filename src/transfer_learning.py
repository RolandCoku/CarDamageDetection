import tensorflow as tf
import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURATION ---
# MUST match what you just changed in preprocessing.py
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 20


def run_transfer_learning():
    print(f"Loading Data ({IMG_HEIGHT}x{IMG_WIDTH})...")
    X_train, y_train = preprocessing.load_data('../data/training')
    X_val, y_val = preprocessing.load_data('../data/validation')

    print("\nDownloading MobileNetV2 (Pre-trained Brain)...")

    # 1. Load the Pre-trained Model (MobileNetV2)
    # include_top=False means we remove the final layer (which classifies cats/dogs/etc)
    # weights='imagenet' means we load the knowledge learned from millions of images
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # 2. Freeze the Base Model
    # We don't want to destroy the pre-trained knowledge, so we "freeze" these layers.
    base_model.trainable = False

    # 3. Add our own Classifier on top
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Data Augmentation (still useful!)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),

        # The Pre-trained Base
        base_model,

        # This converts the complex features into a single vector
        tf.keras.layers.GlobalAveragePooling2D(),

        # Our custom layers for Car Damage
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Lower learning rate is better for Transfer Learning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining with Transfer Learning...")

    # Callbacks (Early Stopping + Save Best)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_transfer_model.keras', save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Evaluate
    predictions_prob = model.predict(X_val)
    predictions = (predictions_prob > 0.5).astype("int32")

    acc = accuracy_score(y_val, predictions)
    print(f"\n--- Transfer Learning Accuracy: {acc * 100:.2f}% ---")

    cm = confusion_matrix(y_val, predictions)
    print(cm)


if __name__ == "__main__":
    run_transfer_learning()