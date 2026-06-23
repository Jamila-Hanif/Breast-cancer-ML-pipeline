# Breast Cancer Prediction with Machine Learning

Comparative ML pipeline predicting tumor malignancy using the Wisconsin Diagnostic Breast Cancer dataset. Built for MSc dissertation with focus on exploratory data analysis, preprocessing, model comparison, and Explainable AI for clinical trust.

## Project Objective

Predict malignant vs benign tumors from fine needle aspirate cell nucleus features.
Optimized 5 ML models for low False Negative Rate. Added SHAP to make predictions interpretable for clinical use.

## Notebook and Dissertation

Google Colab: https://colab.research.google.com/drive/16lep2x0Uh29Hop47VSn2MAUs8-UQ4wXM - Permission needed 
MSc Dissertation PDF: See `dissertation.pdf` in repository root

The notebook contains full EDA, modeling, evaluation, and SHAP analysis with outputs.

## Dataset

- **Source**: UCI ML Repository / Kaggle - Breast Cancer Wisconsin Diagnostic
- **Size**: 569 samples, 30 numerical features
- **Features**: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension. Each as mean, standard error, worst
- **Target**: Diagnosis - Benign [0] vs Malignant [1]
- **Class Imbalance**: 62.7% Benign, 37.3% Malignant. Addressed via class_weight='balanced' and Recall-focused metrics

## Tech Stack

| Tool/Library | Purpose |
| --- | --- |
| Python 3.10 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | SVM, Random Forest, KNN, Logistic Regression, MLPClassifier |
| Matplotlib, Seaborn | Visualization and correlation heatmaps |
| SciPy | Hierarchical clustering, multicollinearity analysis |
| SHAP | Model explainability and feature impact |
| Kagglehub | Reproducible dataset loading |

## Methodology

### 1. Data Preparation
- Loaded dataset via kagglehub for reproducibility
- Dropped id and Unnamed: 32 empty column
- Label encoded diagnosis: Malignant=1, Benign=0
- Scaled features with StandardScaler for SVM and KNN distance models

### 2. Exploratory Data Analysis and Feature Analysis
- Visualized class distribution: 357 Benign vs 212 Malignant
- Identified multicollinearity: radius_mean, perimeter_mean, area_mean R squared > 0.97
- Applied hierarchical clustering to flag 10 feature pairs with correlation > 0.95 for potential removal

### 3. Modeling
Trained 5 models with class_weight='balanced' and GridSearchCV for hyperparameter tuning:
1. SVM - RBF kernel, tuned C and gamma parameters
2. Random Forest - 100 estimators
3. KNN - k=5, distance weighted
4. Logistic Regression - L2 regularization
5. MLPClassifier - 2 hidden layers

### 4. Explainability with SHAP
Added SHAP to explain model decisions beyond accuracy:
- Global interpretability: SHAP Summary Plot shows features driving predictions across all patients
- Local interpretability: Force plots explain individual predictions
- Validation: Compared SHAP results with Random Forest feature importance for consistency

### 5. Evaluation Metrics
Prioritized Recall, F1-Score, ROC-AUC, and False Negative Rate over Accuracy due to medical cost of missed malignant cases.

## Results

| Model | Accuracy | ROC-AUC | Recall | F1-Score | False Negative Rate |
| --- | --- | --- | --- | --- | --- |
| SVM | 97.19% | 99.58% | 96.23% | 97.00% | 3.77% |
| Random Forest | 96.50% | 99.10% | 91.98% | 96.40% | 8.02% |
| MLPClassifier | 96.80% | 98.90% | 95.75% | 96.10% | 4.25% |
| Logistic Regression | 96.10% | 98.70% | 95.75% | 95.50% | 4.25% |
| KNN | 94.30% | 97.20% | 90.57% | 93.20% | 9.43% |

**Clinical Decision**: SVM selected with 3.77% False Negative rate, which equals 2x fewer missed malignant cases compared to Random Forest and KNN above 8%. In healthcare applications, minimizing False Negatives takes priority over maximizing Accuracy.

## Explainable AI Results

<img width="310" height="207" alt="image" src="https://github.com/user-attachments/assets/e967d1f1-22ca-496b-a750-b774791d5573" />

<img width="281" height="207" alt="image" src="https://github.com/user-attachments/assets/b5ce2ba4-82d3-4a95-abe6-b8b53fbadcc2" />


SHAP Summary Plot for SVM: Top 3 drivers for malignancy prediction are perimeter_worst, concave points_mean, and concavity_mean. High feature values increase malignancy risk, which aligns with clinical knowledge of tumor morphology.


Random Forest Feature Importance confirms the same top 3 features as SHAP. Cross-method agreement increases clinical trust in the model.

**Impact**:
1. Model is interpretable rather than a black box, allowing clinicians to validate logic
2. Top features align with medical literature on tumor characteristics
3. Logistic Regression and Random Forest offer strongest interpretability, while MLPClassifier offers high accuracy with lower explainability

## Key Learnings

1. Safety over Accuracy: Optimized for False Negative Rate due to clinical risk
2. Explainability: SHAP makes machine learning trustworthy for high-stakes decisions
3. Multicollinearity: Hierarchical clustering identified redundant size features before modeling
4. Reproducibility: Kagglehub and standardized pipeline ensure full replication.

## Author

Jamila Hanif
MSc Data Analytics | Data Analyst
Dissertation: Predicting Breast Cancer Using Machine Learning: Comparative Evaluation of SVM, RF, KNN, LR, and ANN Models
LinkedIn: https://www.linkedin.com/in/jamila-h-6206b4194
GitHub: https://github.com/Jamila-Hanif

## License

Educational and academic use only. 
Dataset: Breast Cancer Wisconsin Diagnostic Database, UCI Machine Learning Repository.
This project is for MSc dissertation and portfolio purposes. Not intended for clinical diagnosis.
