import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, confusion_matrix
import preprocessing  # This imports your previous script

# --- CONFIGURATION ---
IMG_HEIGHT = 64
IMG_WIDTH = 64
EPOCHS = 10  # How many times the Neural Network sees the data


def calculate_metrics(model_name, y_true, y_pred):
    """
    Calculates and prints the specific metrics required by the coursework.
    """
    print(f"\n--- Results for {model_name} ---")

    # 1. Correctly/Incorrectly Classified
    cm = confusion_matrix(y_true, y_pred)
    correct = np.trace(cm)
    incorrect = np.sum(cm) - correct
    print(f"Correctly Classified Instances: {correct}")
    print(f"Incorrectly Classified Instances: {incorrect}")

    # 2. Kappa Statistic
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Kappa Statistic: {kappa:.4f}")

    # 3. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: {mae:.4f}")

    # 4. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Accuracy (General)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    return acc


def run_training():
    # 1. Load Data
    print("Loading Data...")
    # Adjust these paths if your folders are named differently in 'data/'
    X_train, y_train = preprocessing.load_data('../data/training')
    X_val, y_val = preprocessing.load_data('../data/validation')

    # 2. Prepare Data for Standard Models (SVM, Random Forest)
    # We must "flatten" the images from (64, 64, 3) to (12288)
    num_pixels = IMG_HEIGHT * IMG_WIDTH * 3
    X_train_flat = X_train.reshape(X_train.shape[0], num_pixels)
    X_val_flat = X_val.reshape(X_val.shape[0], num_pixels)

    # --- MODEL 1: Support Vector Machine (SVM) ---
    print("\nTraining Model 1: Support Vector Machine (SVM)...")
    svm = SVC(kernel='rbf')  # RBF is a standard, powerful kernel
    svm.fit(X_train_flat, y_train)

    # Predict and Evaluate
    predictions_svm = svm.predict(X_val_flat)
    calculate_metrics("SVM", y_val, predictions_svm)

    # --- MODEL 2: Random Forest ---
    print("\nTraining Model 2: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100)  # 100 trees
    rf.fit(X_train_flat, y_train)

    # Predict and Evaluate
    predictions_rf = rf.predict(X_val_flat)
    calculate_metrics("Random Forest", y_val, predictions_rf)

    # --- MODEL 3: Neural Network (CNN) ---
    print("\nTraining Model 3: Neural Network (CNN)...")

    # Build the CNN architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification (0 or 1)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))

    # Predict (CNN gives probabilities, so we round them to 0 or 1)
    predictions_cnn_prob = model.predict(X_val)
    predictions_cnn = (predictions_cnn_prob > 0.5).astype("int32")

    calculate_metrics("Neural Network (CNN)", y_val, predictions_cnn)


if __name__ == "__main__":
    run_training()