import os

from pyexpat import features

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
from datetime import datetime
from data_preprocessing import load_data, preprocess_data, splitData
from tcn_model import build_tcnModel

#config
dataPath = "C:/Users/ausus/Desktop/Coding Stuff/VSCode/Thesis/TCN/data/CSE-CIC-IDS-2018-V2.csv"
sample_frac = 0.1 #10% of the data for prototyping
epoch = 15
batch_size = 120
data_dir = "data"
model_dir = "models"
results_dir = "training_results"

def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Starting training with sample fraction: {sample_frac}")

    os.makedirs(data_dir, exist_ok = True)
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    print("\n===== Loading and preprocessing data =====")
    df = load_data(dataPath, sample_frac = sample_frac)
    print(f"Loaded dataset with {len(df)} samples")

    features, labels, label_encoder, scaler = preprocess_data(df)
    print(f"Preprocessed data shape: {features.shape}")
    print(f"Class distribution: {pd.Series(labels).value_counts()}")

    features_train, features_val, features_test, labels_train, labels_val, labels_test = splitData(features, labels)
    print(f"\nData splits:")
    print(f"  Train: {features_train.shape[0]} samples")
    print(f"  Val:   {features_val.shape[0]} samples")
    print(f"  Test:  {features_test.shape[0]} samples")

    print("\nSaving preprocessed data...")
    np.save(os.path.join(data_dir, "X_train.npy"), features_train)
    np.save(os.path.join(data_dir, "X_val.npy"), features_val)
    np.save(os.path.join(data_dir, "X_test.npy"), features_test)
    np.save(os.path.join(data_dir, "y_train.npy"), labels_train)
    np.save(os.path.join(data_dir, "y_val.npy"), labels_val)
    np.save(os.path.join(data_dir, "y_test.npy"), labels_test)

    joblib.dump(scaler, os.path.join(data_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(data_dir, "label_encoder.joblib"))

    class_names = label_encoder.classes_
    class_distribution = pd.Series(labels).value_counts()
    class_distribution.index = [class_names[i] for i in class_distribution.index]
    print("\nClass distribution:")
    print(class_distribution)

    #build the model
    print("\n===== Building TCN Model =====")
    model = build_tcnModel(
        input_shape=(features_train.shape[1], features_train.shape[2]),
        num_classes=len(class_names)
    )
    model.summary()

    # Callbacks
    model_path = os.path.join(model_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]

    #train
    print("\n===== Training Model =====")
    print(f"Training for {epoch} epochs with batch size {batch_size}")

    history = model.fit(
        features_train, labels_train,
        validation_data = (features_val, labels_val),
        epochs = epoch,
        batch_size = batch_size,
        callbacks = callbacks,
        verbose = 1
    )

    #save the final model
    final_modelPath = os.path.join(model_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    model.save(final_modelPath)
    print(f"\nFinal model save to {final_modelPath}")

    print("\nSaving training results...")
    save_trainingHistory(history, results_dir)

    val_loss, val_acc = model.evaluate(features_val, labels_val, verbose = 0)
    print(f"\nValidation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

    summary = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": dataPath,
        "sample_fraction": sample_frac,
        "num_samples": len(df),
        "train_samples": len(features_train),
        "val_samples": len(features_val),
        "test_samples": len(features_test),
        "input_shape": features_train.shape[1:],
        "num_classes": len(class_names),
        "epochs": epoch,
        "batch_size": batch_size,
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "best_model_path": model_path,
        "final_model_path": final_modelPath,
        "class_distribution": class_distribution.to_dict(),
        "class_names": class_names.tolist()
    }

    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to {summary_path}")
    print("\nTraining completed successfully!")


def save_trainingHistory(history, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_path = os.path.join(output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plot_path = os.path.join(output_dir, f"learning_curves_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Training history saved to {history_path}")
    print(f"Learning curves saved to {plot_path}")

    return history_path, plot_path

if __name__ == "__main__":
    main()
