# 🐶 Dogs vs. Cats — CNN Classifier & FastAPI Inference

A simple, production-ready pipeline to train a CNN that classifies dog/cat images and to serve predictions via a FastAPI endpoint and a minimal web UI.

---

## Features

- TensorFlow/Keras CNN trained from scratch (BatchNorm + Dropout).
- Automatic best nodel saving via `ModelCheckpoint` (`best_model.keras`).
- FASTAPI server for inference: `/predict` (multipart image upload).
- Lightweight web UI (HTML/CSS/JS) for drag-and-drop image testing.
- Reproducible training with `train/val` split and data cleaning.

---

## Project Structure

Dogs-vs-cats-AI-model/
├─ app.py                  # FastAPI app with / and /predict
├─ model.py                # Data prep + training script
├─ best_model.keras        # Saved best model (created after training)
├─ requirements.txt        # Python dependencies
├─ index.html              # Minimal web interface
└─ static/
├─ style.css
└─ app.js

The dataset is expected under `PetImages/` with subfolders `Cat/` and `Dog/` (Kaggle “Cat and Dog” style). Adjust paths in `model.py` if needed.

---

## Getting Started

### 1) Environment 

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2) Train the Model

```bash
python model.py

what is does:
    •	Scans PetImages/, removes corrupt files, builds a dataframe.
	•	Splits into train/validation using train_test_split.
	•	Trains a small CNN with BatchNormalization and Dropout.
	•	Saves the best checkpoint to best_model.keras (monitors val_loss).
	•	Prints learning curves and best validation accuracy.

### 3) Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

open:
    •	Web UI: http://localhost:8000 (upload an image and see the prediction)
	•	Interactive docs: http://localhost:8000/docs 


## API

POST / predict:
    •	Body: multipart/form-data with field file (image: png/jpg/jpeg).
	•	Response:
        ```json
        {
        "label": "cat",
        "confidence": 0.9823
        }


## Model Details

•	Input size: 128×128 RGB.
•	Backbone: 4× Conv2D blocks (16→128 filters), each with BN + MaxPool.
•	Head: Flatten → Dense(512, relu) + BN + Dropout(0.4) → Dense(1, sigmoid).
•	Optimizer: Adam (lr=3e-4, clipnorm=1.0).
•	Callbacks: ModelCheckpoint(save_best_only=True), ReduceLROnPlateau, EarlyStopping(restore_best_weights=True).


## Results (example)

- Train accuracy: ~93%
- Validation accuracy: ~90%
- Improvement: +4% after adding BatchNorm & Dropout

> Your exact numbers may vary depending on dataset cleaning, augmentations, and random seed.

---

## Configuration

Key knobs (edit in `model.py`):
- Data Augmentation: rotation/zoom/shear/flip in `ImageDataGenerator`.
- Batch size: default `64`.
- Epochs: default `15` with early stopping.
- Learning rate schedule: `ReduceLROnPlateau` halves LR on plateaus.

---

## Reproducibility

- Fixed `random_state=42` for splits.
- Set `shuffle=False` on validation iterator for deterministic evaluation.
- For strict reproducibility, also set global seeds for NumPy/TensorFlow and disable nondeterministic ops.

---

## Troubleshooting

- `OSError: cannot identify image file`**  
  Corrupted images — this repo verifies and filters them before training.

- Different results across runs
  Expected with stochastic optimizers. Use seeds and keep `val_iterator` deterministic.

- `best_model.keras` not created  
  Ensure `ModelCheckpoint(save_best_only=True, monitor='val_loss')` is active and you ran `model.py` to completion.

---

## Roadmap

- Add proper test split (train/val/test).
- Export to TensorFlow Lite for mobile.
- Add Grad-CAM visualization for interpretability.

---

## Contributing

PRs and issues are welcome. Please format code with `black` / `ruff` and include a brief description of changes and validation steps.

---

## Acknowledgements

- Dataset layout compatible with Kaggle “Cats and Dogs”.
- Built with TensorFlow/Keras and FastAPI.