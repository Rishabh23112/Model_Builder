# 🚀 AutoML Tabular Model Builder

AutoML Tabular Model Builder is a machine learning pipeline that automatically analyzes a dataset, preprocesses it, trains multiple models, evaluates their performance, and selects the best model.

The goal of this project is to simplify machine learning workflows for **tabular datasets** by automating the full ML pipeline.

---

# 📌 Features

- Automatic dataset analysis
- Automatic data preprocessing
- Dynamic neural network generation using PyTorch
- GPU-supported training
- Model evaluation with scikit-learn metrics
- Automatic model comparison
- Best model selection

The system compares the performance of:

- Neural Network
- Logistic Regression
- Random Forest

and returns the best-performing model.

---

# 🧠 How It Works

```
CSV Dataset
     ↓
Dataset Analyzer
     ↓
Automatic Preprocessing
     ↓
Neural Network Generator
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Best Model Selection
```

---



# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Rishabh23112/Model_Builder.git
```

Move into the project directory

```bash
cd Model_Builder
```

Install required libraries

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Main libraries used:

- torch
- pandas
- numpy
- scikit-learn

Example `requirements.txt`

```
torch
pandas
numpy
scikit-learn
```

---

# ▶️ Usage

Run the pipeline with a dataset:

```bash
python main.py dataset.csv
```

Example:

```bash
python main.py datasets/sample_dataset.csv
```

---

# 📊 Example Output

```
Dataset Info
-------------------
Samples: 5000
Features: 10
Task: classification

Epoch 1, Loss: 1.21
Epoch 2, Loss: 0.84
Epoch 3, Loss: 0.63

Test Accuracy: 0.82

Model Comparison
----------------------
Neural Network: 0.82
Logistic Regression: 0.78
Random Forest: 0.86

Best Model: Random Forest
```

---

# 🧪 Supported Tasks

| Task | Status |
|-----|------|
| Classification | Supported |
| Regression | Supported |
| GPU Training | Supported |
| Automatic Preprocessing | Supported |

---

# 🛠 Technologies Used

- PyTorch
- scikit-learn
- pandas
- NumPy

---

