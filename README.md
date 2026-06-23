# Predicting Breast Cancer using Machine Learning: Data Analysis Pipeline

## Project Overview
This project forms the technical foundation for my Master's dissertation, *"Predicting Breast Cancer Using Machine Learning: A Comparative Evaluation of SVM, RF, KNN, LR, and ANN Models"*. It focuses on the exploratory data analysis and preprocessing of the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, preparing it for the training and evaluation of multiple machine learning algorithms.

The goal was to understand data characteristics, identify potential issues such as multicollinearity and class imbalance, and establish a clean, reproducible workflow to ensure reliable model performance.

## Dataset
- **Source**: UCI Machine Learning Repository / Kaggle (Breast Cancer Wisconsin Data)
- **Size**: 569 samples, 30 numerical features derived from digitized images of fine needle aspirates (FNA) of breast masses.
- **Features**: Characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension. Each feature is provided as mean value, standard error, and "worst" (mean of the largest 3 values).
- **Target Variable**: Diagnosis (Benign = 0, Malignant = 1)
- **Class Distribution**:
  - Benign: 357 samples (62.7%)
  - Malignant: 212 samples (37.3%)
  - *Note: The dataset is moderately imbalanced, which was addressed during model training using appropriate techniques.*

## Technologies & Libraries
| Tool/Library | Purpose |
| :--- | :--- |
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and numerical operations |
| Matplotlib & Seaborn | Data visualization and statistical plotting |
| SciPy | Hierarchical clustering and distance calculations |
| Kagglehub | Reproducible dataset downloading |

## Workflow & Methodology

### 1. Data Loading & Preparation
- Loaded the dataset directly via `kagglehub` to ensure reproducibility.
- Removed irrelevant columns (`id`, empty columns) that do not contribute to prediction.
- Encoded the categorical diagnosis label into numerical values for compatibility with machine learning algorithms.

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: Visualized the split between benign and malignant cases to identify imbalance.
- **Statistical Summary**: Generated descriptive statistics to understand feature ranges and distributions.
- **Correlation Analysis**:
  - Created standard and clustered correlation matrices to examine relationships between features.
  - Identified strong multicollinearity, particularly among size-related features (e.g., `radius_mean`, `perimeter_mean`, and `area_mean` showed R² values > 0.97).
  - Used hierarchical clustering to group similar features and highlighted pairs with correlation coefficients > 0.95, flagging them for potential feature selection or dimensionality reduction.

### 3. Key Insights & Decisions
- **Feature Redundancy**: High correlation among size features means they carry similar information. Retaining all can inflate variance in models like KNN or Linear Regression; therefore, representative features were selected or scaling was applied accordingly.
- **Class Imbalance**: Recognized that overall accuracy can be misleading. Consequently, model evaluation prioritized metrics like Recall (Sensitivity), F1-Score, and ROC-AUC, and techniques like class weighting were used during training.
- **Interpretability**: The analysis was designed with clinical relevance in mind, ensuring that features used in models align with known medical properties of tumor cells.

## Results & Impact
This pipeline provided the clean, validated dataset used in the subsequent comparative study of machine learning models. The rigorous preparation contributed to strong performance across all algorithms:
- **Best Performing Model**: Support Vector Machine (SVM) achieved 97.19% accuracy, 99.58% ROC-AUC, and the lowest False Negative Rate (3.77%) — a critical metric for medical diagnosis.
- **Other Models**: Artificial Neural Networks (ANN) and Logistic Regression (LR) also performed excellently, offering different trade-offs between accuracy and interpretability.

## Future Improvements
- Extend the codebase to include the full model training, tuning, and evaluation logic.
- Implement Explainable AI techniques (SHAP/LIME) to visualize feature impact on predictions.
- Test model generalizability on external datasets to move beyond the benchmark WDBC data.
- Integrate solutions for class imbalance (e.g., SMOTE) to compare performance improvements.

## Author
Jamila Hanif
- MSc Data Analytics 
- This work was completed as part of a Master's dissertation.

## License
This project is available for educational and academic purposes. Please cite or reference if you use parts of this code.
