import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import preprocessing

# --- CONFIGURATION ---
IMG_HEIGHT = 64  # Keep small for speed since we run this 5 times
IMG_WIDTH = 64
N_FOLDS = 5  # Number of splits (Standard is 5 or 10)
EPOCHS = 15  # Fewer epochs per fold to keep total time reasonable


def get_model():
    """
    Returns a FRESH, compiled model every time we call it.
    We must reset the brain for every fold!
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # Data Augmentation
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),

        # CNN Architecture
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
    return model


def run_k_fold():
    print("Loading ALL Data for Cross-Validation...")

    # 1. Load both folders
    X_train_part, y_train_part = preprocessing.load_data('../data/training')
    X_val_part, y_val_part = preprocessing.load_data('../data/validation')

    # 2. Combine them into one big dataset
    X = np.concatenate((X_train_part, X_val_part), axis=0)
    y = np.concatenate((y_train_part, y_val_part), axis=0)

    print(f"\nTotal Dataset Size: {len(X)} images")
    print(f"Starting {N_FOLDS}-Fold Cross-Validation...\n")

    # 3. Define K-Fold Splitter
    # StratifiedKFold ensures each fold has the same % of damaged cars
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []

    fold_no = 1
    for train_index, test_index in skf.split(X, y):
        print(f"--- Training Fold {fold_no} / {N_FOLDS} ---")

        # Split Data for this specific fold
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Get a fresh model
        model = get_model()

        # Train (Silent mode verbose=0 to reduce clutter, showing only results)
        model.fit(X_train_fold, y_train_fold, epochs=EPOCHS, verbose=0)

        # Evaluate
        predictions_prob = model.predict(X_test_fold, verbose=0)
        predictions = (predictions_prob > 0.5).astype("int32")
        acc = accuracy_score(y_test_fold, predictions)

        print(f"Fold {fold_no} Accuracy: {acc * 100:.2f}%")
        fold_accuracies.append(acc)
        fold_no += 1

    # 4. Final Results
    print("\n" + "=" * 30)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 30)
    print(f"Individual Scores: {[f'{score * 100:.2f}%' for score in fold_accuracies]}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(fold_accuracies) * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    run_k_fold()