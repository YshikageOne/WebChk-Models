from tabnanny import verbose

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import json
from datetime import datetime


def evaluate_model(model_path, features_test, labels_test, label_encoder = None, output_dir = "results"):
    #output directory
    os.makedirs(output_dir, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model on test set....")
    test_loss, test_acc = model.evaluate(features_test, labels_test, verbose = 0)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    #generate predictions
    labels_pred = model.predict(features_test)
    labels_pred_classes = np.argmax(labels_pred, axis = 1)

    if label_encoder:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels_test)))]

    #classification report
    clf_report = classification_report(
        labels_test,
        labels_pred_classes,
        target_names = class_names,
        output_dict = True
    )

    #save the report
    report_df = pd.DataFrame(clf_report).transpose()
    report_path = os.path.join(output_dir, f"classification_report_{timestamp}.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    #confusion matrix
    confusionMatrix = confusion_matrix(labels_test, labels_pred_classes)
    confusionMatrix_df = pd.DataFrame(confusionMatrix, index = class_names, columns = class_names)

    #save the matrix
    confusionMatrix_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.csv")
    confusionMatrix_df.to_csv(confusionMatrix_path)
    print(f"Confusion matrix saved to {confusionMatrix_path}")

    #plot
    plt.figure(figsize = (12, 10))
    sns.heatmap(confusionMatrix_df, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.close()

    #save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "evaluation": timestamp,
        "num_test_samples": int(len(features_test)),
        "num_classes": int(len(class_names)),
        "classification_report_path": report_path,
        "confusion_matrix_path": confusionMatrix_path,
        "confusion_matrix_plot": plot_path
    }

    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent = 2)

    print(f"Full metrics saved to {metrics_path}")

    return metrics

def load_testData(data_dir = 'data'):
    print("Loading test data and preprocessing objects...")
    features_test = np.load(os.path.join(data_dir, "X_test.npy"))
    labels_test = np.load(os.path.join(data_dir, "y_test.npy"))
    scaler = joblib.load(os.path.join(data_dir, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(data_dir, "label_encoder.joblib"))
    return features_test, labels_test, label_encoder


if __name__ == "__main__":
    #config
    MODEL_PATH = "best_model.h5"
    DATA_DIR = "data"
    OUTPUT_DIR = "evaluation_results"

    # Load test data
    features_test, labels_test, label_encoder = load_testData(DATA_DIR)

    # Evaluate model
    metrics = evaluate_model(
        MODEL_PATH,
        features_test,
        labels_test,
        label_encoder,
        OUTPUT_DIR
    )

    # Print summary
    print("\nEvaluation Complete!")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Number of test samples: {metrics['num_test_samples']}")
    print(f"Number of classes: {metrics['num_classes']}")