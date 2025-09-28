
# Machine Learning Lab: kNN and Decision Trees

## 📚 Overview
This repository contains implementations of k-Nearest Neighbors (kNN) and Decision Tree algorithms using scikit-learn, along with comprehensive performance evaluation metrics on the Iris dataset.

## 🎯 Objectives
- Implement and analyze Decision Trees with varying complexity
- Explore kNN classification with different k values
- Visualize decision boundaries
- Evaluate model performance using various metrics
- Understand underfitting vs overfitting

## 📂 Repository Structure
```
│
├── README.md
├── requirements.txt
├── CS-5710_11595_Homework2.ipynb
```

## 🔧 Requirements
```python
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
seaborn==0.11.1
scikit-learn==0.24.2
```

## 💻 Installation

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running in Google Colab
1. Open Google Colab: https://colab.research.google.com/
2. Upload the notebook files
3. Run cells sequentially

### Running Locally
```bash
jupyter notebook
# Navigate to notebooks folder and open desired notebook
```

## 📊 Experiments

### Q7: Decision Tree Classification
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Max Depth Values**: 1, 2, 3
- **Key Finding**: Depth=3 achieves best balance between bias and variance

**Results:**
| Max Depth | Train Accuracy | Test Accuracy |
|-----------|---------------|---------------|
| 1         | ~0.67         | ~0.63         |
| 2         | ~0.97         | ~0.97         |
| 3         | ~0.99         | ~0.97         |

### Q8: kNN Classification
- **Features Used**: Sepal Length and Sepal Width (2D visualization)
- **k Values**: 1, 3, 5, 10
- **Key Finding**: Higher k values create smoother decision boundaries

**Boundary Characteristics:**
- k=1: Most complex, prone to overfitting
- k=3: Balanced complexity
- k=5: Good generalization
- k=10: Risk of underfitting

### Q9: Performance Evaluation
- **Model**: kNN with k=5
- **Features**: All 4 features
- **Test Accuracy**: 100% (on this specific split)

**Metrics Summary:**
- Accuracy: 1.00
- Precision: 1.00 (all classes)
- Recall: 1.00 (all classes)
- F1-Score: 1.00 (all classes)

## 📈 Key Visualizations

### Decision Boundaries (kNN)
Shows how decision regions change with different k values:
- k=1: Jagged, complex boundaries
- k=3: Moderately smooth boundaries
- k=5: Balanced smoothness
- k=10: Very smooth, simple boundaries

### Confusion Matrix
Perfect classification achieved on test set:
```
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]
```

### ROC Curves
- Setosa: AUC = 1.00
- Versicolor: AUC = 1.00
- Virginica: AUC = 1.00

## 🔍 Key Insights

### 1. Underfitting vs Overfitting
- **Underfitting** (Depth=1): Both training and test accuracy are low (~67%)
- **Good fit** (Depth=2): High accuracy on both sets (~97%)
- **Potential Overfitting** (Depth=3): Very high training accuracy (99%) with slightly lower test accuracy

### 2. kNN Behavior
- **k=1**: Each point creates its own region (memorization)
- **k=3-5**: Balanced between local and global patterns
- **k=10**: Considers broader neighborhood (smoothing)

### 3. Iris Dataset Characteristics
- Exceptionally clean and well-separated
- Three distinct species with measurable differences
- Perfect classification is achievable with proper algorithms

## ⚠️ Important Notes

- The 100% accuracy on Iris is not typical for real-world datasets
- This exceptional performance is due to:
  - Clean, noise-free data
  - Well-separated classes
  - Small, balanced dataset
  - Optimal algorithm choice (kNN with k=5)
- Real-world datasets typically have:
  - Noise and outliers
  - Overlapping classes
  - Missing values
  - Higher dimensionality

## 📖 Theory Background

### k-Nearest Neighbors (kNN)
```
Algorithm:
1. Calculate distance from test point to all training points
2. Select k nearest neighbors
3. Assign class by majority vote
```

**Advantages:**
- Simple and intuitive
- No training phase
- Can capture complex patterns

**Disadvantages:**
- Computationally expensive at prediction time
- Sensitive to feature scaling
- Curse of dimensionality

### Decision Trees
```
Algorithm:
1. Select best feature to split on (using information gain/Gini impurity)
2. Create branches for each value
3. Repeat recursively until stopping criterion met
```

**Advantages:**
- Interpretable
- Handles non-linear patterns
- No feature scaling needed

**Disadvantages:**
- Prone to overfitting
- Unstable (small changes can result in different trees)
- Biased towards features with more levels

## 📊 Code Examples

### Basic kNN Implementation
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
```

### Basic Decision Tree Implementation
```python
from sklearn.tree import DecisionTreeClassifier

# Train model
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Predict
predictions = dt.predict(X_test)
```

## 🤝 Contributing
Feel free to open issues or submit pull requests for improvements.

## 📝 License
MIT License - Use freely for educational purposes

## 👥 Author
Hemanth Vamsi Krishna Devadula

## 🙏 Acknowledgments
- scikit-learn documentation and community
- Fisher's Iris dataset (1936)
- UCI Machine Learning Repository
- Course instructors and materials

## 📚 References
1. Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"
2. Altman, N. S. (1992). "An introduction to kernel and nearest-neighbor nonparametric regression"
3. Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32
