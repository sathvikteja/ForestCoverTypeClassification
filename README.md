# Forest Cover Type Classification using Machine Learning and Deep Neural Networks

This repository presents a large-scale multiclass classification study on the
Roosevelt National Forest dataset, aiming to predict forest cover types from
cartographic and environmental features.

## Dataset
- Source: U.S. Forest Service (Roosevelt National Forest)
- Samples: 581,012
- Features: 54 (10 continuous + 44 one-hot encoded categorical)
- Target: 7 forest cover types
- Challenge: Severe class imbalance (minority class ≈ 0.47%)

## Methodology
- Exploratory Data Analysis (EDA) to identify key discriminative features
- Stratified train/validation/test split
- Feature scaling using StandardScaler
- Models implemented:
  - Multinomial Logistic Regression
  - Linear Support Vector Machine
  - Deep Neural Network (PyTorch)

## Deep Neural Network Architecture
- Fully connected MLP: 512 → 256 → 128 → 64 → 7
- Batch Normalization + ReLU activations
- Dropout regularization (0.25)
- Class-weighted CrossEntropy loss to address imbalance
- Optimizer: Adam with learning-rate scheduling and early stopping

## Results
| Model | Test Accuracy | Macro F1 |
|------|--------------|----------|
| Logistic Regression | ~72.5% | ~0.53 |
| Linear SVM | ~71.3% | ~0.46 |
| Deep Neural Network | **87.8%** | **0.82** |

The deep neural network significantly outperforms classical models,
achieving over 93% recall across all classes, including severely underrepresented ones.

## Files
- `mlForestDNN.ipynb` – Model implementation and experiments
- `mlForestReport.pdf` – Detailed technical report

## Tools & Libraries
- Python, NumPy, Pandas
- Scikit-learn
- PyTorch
- Matplotlib / Seaborn

## Author
Sathvik Teja 
