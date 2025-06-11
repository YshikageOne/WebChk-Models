import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, \
    TrainerState, TrainerControl, set_seed, EarlyStoppingCallback
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

#by clyde
#tired

#set up logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:

    def __init__(self, diverseVul_path, vulnCode_path):
        self.diverseVul_path = diverseVul_path
        self.vulnCode_path = vulnCode_path

    def loadDiverseVul(self):
        try:
            with open(self.diverseVul_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            with ThreadPoolExecutor() as executor:
                data = list(executor.map(json.loads, lines))
            return [(item['func'], item['target']) for item in data]
        except Exception as e:
            logger.error(f"Error loading DiverseVul: {e}")
            return []

    def loadVulnCode(self):
        data = []

        def process_file(path):
            try:
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                with ThreadPoolExecutor() as executor:
                    return list(executor.map(json.loads, lines))
            except Exception as e:
                logger.error(f"Error loading Vuln-Code from {path}: {e}")
                return []

        with ThreadPoolExecutor() as executor:
            all_results = executor.map(process_file, self.vulnCode_path)
            for results in all_results:
                if results:
                    data.extend([(item['func'], item['target']) for item in results])
        return data

    def getDatasets(self, test_size = 0.2):
        diverseVul = self.loadDiverseVul()
        vulnCode = self.loadVulnCode()

        all_data = diverseVul + vulnCode
        texts, labels = zip(*all_data)

        return train_test_split(
            texts, labels,
            test_size = test_size,
            stratify = labels,
            random_state = 42
        )


class BERTDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length = 512):
        self.encodings = tokenizer(
            list(texts),
            truncation = True,
            padding = True,
            max_length = max_length,
            return_tensors = "pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

class MetricsCallback(TrainerCallback):

    def __init__(self):
        self.metrics = []

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get('metrics', {})
        self.metrics.append(metrics)

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.startTime = None
        self.epoch_startTime = None
        self.metrics_log = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.startTime = time.time()
        print("Starting Training..")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_startTime = time.time()
        print(f"\nEpoch {state.epoch + 1} / {args.num_train_epochs}")

    def on_log(self, args, state, control, logs = None, **kwargs):
        if logs:
            metrics = {
                "step": state.global_step,
                "loss": logs.get("loss"),
                "lr": logs.get("learning_rate"),
                "epoch": state.epoch,
                "time": f"{(time.time() - self.startTime) / 60:.1f}min"
            }

            print(f"Step {metrics['step']}: Loss = {metrics['loss']:.3f} | LR = {metrics['lr']:.1e} | Time = {metrics['time']}")

            self.metrics_log.append(metrics)

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = state.log_history[-1]
        print(f"\nValidation Metrics (Epoch {state.epoch+1}")
        print(f"  Accuracy:  {metrics.get('eval_accuracy', 0):.3%}")
        print(f"  F1 Score:  {metrics.get('eval_f1', 0):.3%}")
        print(f"  Precision: {metrics.get('eval_precision', 0):.3%}")
        print(f"  Recall:    {metrics.get('eval_recall', 0):.3%}\n")

class BERTTrainer:

    def __init__(self, train_dataset, val_dataset):
        print("Loading BERT base model...")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2,
            ignore_mismatched_sizes = True #fine tuning
        )

        print("BERT model loaded successfully")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.metrics_callback = MetricsCallback()
        self.progress_callback = ProgressCallback()

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)

        return{
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self):

        set_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        training_args = TrainingArguments(
            output_dir = "./results",
            num_train_epochs = 3,
            per_device_train_batch_size = 14,
            per_device_eval_batch_size = 14,
            gradient_accumulation_steps = 2,
            do_train = True,
            do_eval = True,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            logging_dir = "./logs",
            log_level = "info",
            logging_steps = 10,
            load_best_model_at_end = True,
            save_total_limit = 2,
            metric_for_best_model = 'eval_f1',
            report_to = "none",
            fp16 = torch.cuda.is_available(),
            remove_unused_columns = False,
            dataloader_num_workers = 4,
            disable_tqdm = False,
            max_grad_norm = 1.0
        )

        trainer = Trainer(
            model = self.model.to(device),
            args = training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.val_dataset,
            compute_metrics = self.compute_metrics,
            callbacks = [self.metrics_callback, self.progress_callback, EarlyStoppingCallback(early_stopping_patience = 2)]
        )

        trainer.train()
        return trainer

def plot_training_progress(callback):
    history = callback.metrics_log

    steps = [m['step'] for m in history if 'loss' in m]
    losses = [m['loss'] for m in history if 'loss' in m]
    times = [float(m['time'].split('min')[0]) for m in history if 'time' in m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(steps, losses, 'b-o')
    ax1.set_title('Training Loss Over Steps')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(times, losses, 'g-o')
    ax2.set_title('Training Loss Over Time')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()


def plot_training_metrics(history, output_path = "training_metrics.png"):
    epochs = range(1, len(history.metrics) + 1)

    train_losses = [m.get("loss", 0) for m in history.metrics]
    eval_accuracies = [m.get("eval_accuracy", 0) for m in history.metrics]
    eval_f1s = [m.get("eval_f1", 0) for m in history.metrics]
    eval_precisions = [m.get("eval_precision", 0) for m in history.metrics]
    eval_recalls = [m.get("eval_recall", 0) for m in history.metrics]

    fig, axs = plt.subplots(2, 2, figsize = (14,10))
    fig.suptitle('Training Metrics')

    #training loss
    axs[0, 0].plot(epochs, train_losses, 'b-o', label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    #evaluation accuracy
    axs[0, 1].plot(epochs, eval_accuracies, 'g-o', label='Evaluation Accuracy')
    axs[0, 1].set_title('Evaluation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    #f1 scaore
    axs[1, 0].plot(epochs, eval_f1s, 'r-o', label='Evaluation F1 Score')
    axs[1, 0].set_title('Evaluation F1 Score')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    #precision and recall
    axs[1, 1].plot(epochs, eval_precisions, 'm-o', label='Precision')
    axs[1, 1].plot(epochs, eval_recalls, 'c-o', label='Recall')
    axs[1, 1].set_title('Evaluation Precision & Recall')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()
    axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_metrics_summary(history, output_path = "training_summary.json"):
    summary = {
        "final_metrics": {
            "accuracy": float(history.metrics[-1]["eval_accuracy"]),
            "f1": float(history.metrics[-1]["eval_f1"]),
            "precision": float(history.metrics[-1]["eval_precision"]),
            "recall": float(history.metrics[-1]["eval_recall"])
        },
        "best_metrics": {
            "accuracy": float(max(m["eval_accuracy"] for m in history.metrics)),
            "f1": float(max(m["eval_f1"] for m in history.metrics)),
            "precision": float(max(m["eval_precision"] for m in history.metrics)),
            "recall": float(max(m["eval_recall"] for m in history.metrics))
        },
        "history": [
            {
                "epoch": i + 1,
                "loss": float(m["loss"]),
                "accuracy": float(m["eval_accuracy"]),
                "f1": float(m["eval_f1"]),
                "precision": float(m["eval_precision"]),
                "recall": float(m["eval_recall"])
            }
            for i, m in enumerate(history.metrics)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent = 2)

def main():
    print(f"Current working directory: {os.getcwd()}")
    print("Loading datasets...")

    #datasets
    diversevul_path = "data/DiverseVul/diversevul_20230702.json"
    vuln_code_paths = [
        "data/Vuln-Code/primevul_train.jsonl",
        "data/Vuln-Code/primevul_valid.jsonl"
    ]

    #loading and prepping the datasets
    loader = DatasetLoader(diversevul_path, vuln_code_paths)
    print("Loading Data")
    train_texts, val_texts, train_labels, val_labels = loader.getDatasets()

    print(f"\nLoaded {len(train_texts)} training samples")
    print(f"Loaded {len(val_texts)} validation samples")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = BERTDataset(train_texts, train_labels, tokenizer)
    val_dataset = BERTDataset(val_texts, val_labels, tokenizer)

    #train the model
    trainer = BERTTrainer(train_dataset, val_dataset)
    trainer = trainer.train()

    metrics_history = trainer.metrics_callback
    plot_training_metrics(metrics_history)
    plot_training_progress(trainer.progress_callback)
    save_metrics_summary(metrics_history)

    trainer.save_model("./BERT Final Model")
    tokenizer.save_pretrained("./BERT Final Model")

    print("\nTraining Complete")
    print(f"Total Duration: {(time.time() - trainer.progress_callback.start_time) / 60:.1f} minutes")
    print(f"Best Model Saved at: ./BERT Final Model")

if __name__ == "__main__":
    main()