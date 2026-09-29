import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

import config


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "./data"

TRAIN_FILE = os.path.join(DATA_DIR, "trainDataset.pkl")
VAL_FILE = os.path.join(DATA_DIR, "valDataset.pkl")
TEST_FILE = os.path.join(DATA_DIR, "testDataset.pkl")
HALO_FILE = "./results/datasets/haloDataset.pkl"

NUM_LABELS = 25

NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# Reproducibility
# ============================================================

SEED = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load Datasets
# ============================================================

print("Loading datasets...")

with open(TRAIN_FILE, "rb") as f:
    train_dataset = pickle.load(f)

with open(VAL_FILE, "rb") as f:
    val_dataset = pickle.load(f)

with open(TEST_FILE, "rb") as f:
    test_dataset = pickle.load(f)

with open(HALO_FILE, "rb") as f:
    halo_dataset = pickle.load(f)

print(f"Real training records: {len(train_dataset)}")
print(f"Validation records: {len(val_dataset)}")
print(f"Test records: {len(test_dataset)}")
print(f"HALO synthetic records: {len(halo_dataset)}")


# ============================================================
# Filter Invalid / Empty Records
# ============================================================

def valid_patient(patient):

    return (
        "visits" in patient
        and len(patient["visits"]) > 0
        and "labels" in patient
    )


train_dataset = [
    p for p in train_dataset
    if valid_patient(p)
]

val_dataset = [
    p for p in val_dataset
    if valid_patient(p)
]

test_dataset = [
    p for p in test_dataset
    if valid_patient(p)
]

halo_dataset = [
    p for p in halo_dataset
    if valid_patient(p)
]


# ============================================================
# Downstream Diagnosis Model
# ============================================================

class DiagnosisModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Code embedding
        self.embedding = nn.Linear(
            config.code_vocab_size,
            64
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Classification layer
        self.fc = nn.Linear(
            32 * 2,
            1
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # x:
        # [batch, visits, code_vocab_size]

        x = self.embedding(x)

        # [batch, visits, 64]

        output, (hidden, cell) = self.lstm(x)

        # Take final forward + backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        hidden_state = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        output = self.fc(hidden_state)

        output = self.sigmoid(output)

        return output.squeeze(1)


# ============================================================
# Convert Patients to Model Input
# ============================================================

def create_batch(dataset):

    batch_size = len(dataset)

    batch_ehr = np.zeros(
        (
            batch_size,
            config.n_ctx,
            config.code_vocab_size
        ),
        dtype=np.float32
    )

    for i, patient in enumerate(dataset):

        visits = patient["visits"]

        # Limit number of visits to model context
        visits = visits[:config.n_ctx]

        for j, visit in enumerate(visits):

            for code in visit:

                if (
                    0 <= code
                    < config.code_vocab_size
                ):

                    batch_ehr[i, j, code] = 1.0

    return torch.tensor(
        batch_ehr,
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# Get Labels
# ============================================================

def get_labels(dataset, label_index):

    labels = []

    for patient in dataset:

        labels.append(
            patient["labels"][label_index]
        )

    return np.array(
        labels,
        dtype=np.float32
    )


# ============================================================
# Balance Dataset for a Disease Label
# ============================================================

def create_balanced_dataset(
    dataset,
    label_index,
    target_size=5000
):

    positives = []
    negatives = []

    for patient in dataset:

        label = patient["labels"][label_index]

        if label == 1:
            positives.append(patient)
        else:
            negatives.append(patient)

    if len(positives) == 0 or len(negatives) == 0:

        return None

    # Sample equal number of positive and negative examples
    half = target_size // 2

    positive_sample = random.choices(
        positives,
        k=half
    )

    negative_sample = random.choices(
        negatives,
        k=half
    )

    balanced = (
        positive_sample
        + negative_sample
    )

    random.shuffle(balanced)

    return balanced


# ============================================================
# Train Diagnosis Model
# ============================================================

def train_model(
    training_data,
    validation_data,
    label_index
):

    model = DiagnosisModel().to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCELoss()

    best_val_loss = float("inf")

    best_state = None

    for epoch in range(NUM_EPOCHS):

        model.train()

        random.shuffle(training_data)

        total_train_loss = 0
        num_batches = 0

        for start in range(
            0,
            len(training_data),
            BATCH_SIZE
        ):

            batch = training_data[
                start:start + BATCH_SIZE
            ]

            if len(batch) == 0:
                continue

            x = create_batch(batch)

            y = torch.tensor(
                get_labels(
                    batch,
                    label_index
                ),
                dtype=torch.float32,
                device=DEVICE
            )

            optimizer.zero_grad()

            predictions = model(x)

            loss = criterion(
                predictions,
                y
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            num_batches += 1

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0
        validation_batches = 0

        with torch.no_grad():

            for start in range(
                0,
                len(validation_data),
                BATCH_SIZE
            ):

                batch = validation_data[
                    start:start + BATCH_SIZE
                ]

                x = create_batch(batch)

                y = torch.tensor(
                    get_labels(
                        batch,
                        label_index
                    ),
                    dtype=torch.float32,
                    device=DEVICE
                )

                predictions = model(x)

                loss = criterion(
                    predictions,
                    y
                )

                validation_loss += loss.item()

                validation_batches += 1

        avg_train_loss = (
            total_train_loss / num_batches
            if num_batches > 0
            else 0
        )

        avg_val_loss = (
            validation_loss / validation_batches
            if validation_batches > 0
            else 0
        )

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        print(
            f"Label {label_index} | "
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f}"
        )

    if best_state is not None:

        model.load_state_dict(best_state)

    return model


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    model,
    test_data,
    label_index
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for start in range(
            0,
            len(test_data),
            BATCH_SIZE
        ):

            batch = test_data[
                start:start + BATCH_SIZE
            ]

            x = create_batch(batch)

            predictions = model(x)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                get_labels(
                    batch,
                    label_index
                )
            )

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    if len(np.unique(all_labels)) > 1:

        auroc = roc_auc_score(
            all_labels,
            all_predictions
        )

    else:

        auroc = np.nan


    # --------------------------------------------------------
    # AUPRC
    # --------------------------------------------------------

    if np.sum(all_labels) > 0:

        auprc = average_precision_score(
            all_labels,
            all_predictions
        )

    else:

        auprc = np.nan


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    binary_predictions = (
        all_predictions >= 0.5
    ).astype(int)

    f1 = f1_score(
        all_labels,
        binary_predictions,
        zero_division=0
    )

    return auroc, auprc, f1


# ============================================================
# Run Downstream Experiment
# ============================================================

results = {
    "real": [],
    "halo": []
}


for label_index in range(NUM_LABELS):

    print("\n" + "=" * 60)

    print(
        f"DOWNSTREAM TASK - LABEL {label_index + 1}"
    )

    print("=" * 60)


    # ========================================================
    # REAL DATA
    # ========================================================

    print("\nTraining on REAL data...")

    real_training_data = create_balanced_dataset(
        train_dataset,
        label_index
    )

    if real_training_data is None:

        print(
            "Skipping label because it does "
            "not contain both classes."
        )

        continue

    real_model = train_model(
        real_training_data,
        val_dataset,
        label_index
    )

    real_auroc, real_auprc, real_f1 = evaluate_model(
        real_model,
        test_dataset,
        label_index
    )

    print(
        f"REAL | "
        f"AUROC: {real_auroc:.4f} | "
        f"AUPRC: {real_auprc:.4f} | "
        f"F1: {real_f1:.4f}"
    )


    # ========================================================
    # HALO SYNTHETIC DATA
    # ========================================================

    print("\nTraining on HALO synthetic data...")

    halo_training_data = create_balanced_dataset(
        halo_dataset,
        label_index
    )

    if halo_training_data is None:

        print(
            "Skipping HALO label because it does "
            "not contain both classes."
        )

        continue

    halo_model = train_model(
        halo_training_data,
        val_dataset,
        label_index
    )

    halo_auroc, halo_auprc, halo_f1 = evaluate_model(
        halo_model,
        test_dataset,
        label_index
    )

    print(
        f"HALO | "
        f"AUROC: {halo_auroc:.4f} | "
        f"AUPRC: {halo_auprc:.4f} | "
        f"F1: {halo_f1:.4f}"
    )


    # ========================================================
    # Save Results
    # ========================================================

    results["real"].append({
        "label": label_index,
        "AUROC": real_auroc,
        "AUPRC": real_auprc,
        "F1": real_f1
    })

    results["halo"].append({
        "label": label_index,
        "AUROC": halo_auroc,
        "AUPRC": halo_auprc,
        "F1": halo_f1
    })


# ============================================================
# Save Final Results
# ============================================================

os.makedirs(
    "./results/synthetic_training_stats",
    exist_ok=True
)

output_file = (
    "./results/synthetic_training_stats/"
    "halo_downstream_results.pkl"
)

with open(output_file, "wb") as f:

    pickle.dump(
        results,
        f
    )


# ============================================================
# Print Summary
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DOWNSTREAM RESULTS")
print("=" * 70)

print(
    f"{'Dataset':<15}"
    f"{'AUROC':<15}"
    f"{'AUPRC':<15}"
    f"{'F1':<15}"
)

print("-" * 70)


real_results = results["real"]
halo_results = results["halo"]

if len(real_results) > 0:

    real_auroc = np.nanmean([
        x["AUROC"]
        for x in real_results
    ])

    real_auprc = np.nanmean([
        x["AUPRC"]
        for x in real_results
    ])

    real_f1 = np.nanmean([
        x["F1"]
        for x in real_results
    ])

    print(
        f"{'REAL':<15}"
        f"{real_auroc:<15.4f}"
        f"{real_auprc:<15.4f}"
        f"{real_f1:<15.4f}"
    )


if len(halo_results) > 0:

    halo_auroc = np.nanmean([
        x["AUROC"]
        for x in halo_results
    ])

    halo_auprc = np.nanmean([
        x["AUPRC"]
        for x in halo_results
    ])

    halo_f1 = np.nanmean([
        x["F1"]
        for x in halo_results
    ])

    print(
        f"{'HALO':<15}"
        f"{halo_auroc:<15.4f}"
        f"{halo_auprc:<15.4f}"
        f"{halo_f1:<15.4f}"
    )


print("\nResults saved to:")
print(output_file)