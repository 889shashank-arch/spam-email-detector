# Spam Email Detector - Complete Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage Guide](#usage-guide)
5. [Technical Details](#technical-details)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)

---

## Introduction

### Overview
The Spam Email Detector is a comprehensive machine learning project that classifies emails as spam or ham (legitimate) using two popular classification algorithms: Naive Bayes and Logistic Regression.

### Key Features
- ✅ Two classification algorithms (Naive Bayes & Logistic Regression)
- ✅ TF-IDF feature extraction
- ✅ Comprehensive model evaluation
- ✅ Interactive web interface
- ✅ Jupyter notebook for analysis
- ✅ Complete test suite
- ✅ Model persistence
- ✅ Detailed visualizations

### Technologies Used
- **Python 3.8+**
- **scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Flask**: Web framework
- **Pytest**: Testing

---

## Installation

### Prerequisites
```bash
# Check Python version (requires 3.8+)
python --version

# Install pip if not available
python -m ensurepip --upgrade
```

### Setup Steps

1. **Navigate to project directory**
```bash
cd spam_email_detector
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## Project Structure

```
spam_email_detector/
│
├── data/                           # Dataset storage
│   └── (datasets go here)
│
├── models/                         # Saved trained models
│   ├── naive_bayes_model.pkl
│   └── logistic_regression_model.pkl
│
├── notebooks/                      # Jupyter notebooks
│   └── spam_detection_analysis.ipynb
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data preprocessing module
│   ├── feature_extraction.py      # Feature extraction module
│   ├── naive_bayes_classifier.py  # Naive Bayes implementation
│   ├── logistic_regression_classifier.py  # LR implementation
│   ├── model_evaluation.py        # Evaluation metrics
│   └── web_app.py                 # Flask web application
│
├── tests/                         # Unit tests
│   └── test_spam_detector.py
│
├── results/                       # Output visualizations
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── comparison_charts/
│
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── DOCUMENTATION.md              # This file
```

---

## Usage Guide

### Option 1: Run Complete Pipeline

Execute the entire spam detection pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess data
2. Extract TF-IDF features
3. Train both classifiers
4. Evaluate models
5. Generate visualizations
6. Save trained models

### Option 2: Web Interface

Launch the interactive web application:

```bash
python src/web_app.py
```

Then open your browser to `http://localhost:5000`

Features:
- Real-time email classification
- Dual model predictions
- Confidence scores
- Example emails to test

### Option 3: Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/spam_detection_analysis.ipynb
```

### Option 4: Python API

Use the classifiers programmatically:

```python
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.naive_bayes_classifier import NaiveBayesSpamClassifier

# Initialize
preprocessor = DataPreprocessor()
extractor = FeatureExtractor()
classifier = NaiveBayesSpamClassifier()

# Preprocess text
text = "Win a free iPhone now!"
processed = preprocessor.preprocess_text(text)

# Extract features
features = extractor.transform([processed])

# Predict
prediction = classifier.predict(features)
probability = classifier.predict_proba(features)
```

---

## Technical Details

### Data Preprocessing

The preprocessing pipeline includes:

1. **Text Cleaning**
   - Convert to lowercase
   - Remove URLs
   - Remove email addresses
   - Remove special characters
   - Remove extra whitespace

2. **Tokenization**
   - Split text into words
   - Using NLTK's word_tokenize

3. **Stopword Removal**
   - Remove common English words
   - Keep words longer than 2 characters

4. **Lemmatization**
   - Convert words to base form
   - Using WordNetLemmatizer

### Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)**

Formula:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

where:
TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)
IDF(t) = log(Total documents / Documents containing term t)
```

Parameters:
- Max features: 3000
- N-gram range: (1, 2) - unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 0.95

### Naive Bayes Algorithm

**Multinomial Naive Bayes**

Based on Bayes' Theorem:
```
P(spam|email) = P(email|spam) × P(spam) / P(email)
```

Features:
- Laplace smoothing (alpha=1.0)
- Handles class imbalance
- Fast training and prediction
- Probabilistic outputs

### Logistic Regression Algorithm

**Binary Logistic Regression**

Model:
```
P(y=1|x) = 1 / (1 + exp(-(w·x + b)))
```

Features:
- L2 regularization
- Balanced class weights
- L-BFGS optimization
- Decision function available

### Evaluation Metrics

1. **Accuracy**: Overall correctness
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Spam detection accuracy
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: Spam capture rate
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Harmonic mean
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **ROC-AUC**: Area under ROC curve

---

## API Reference

### DataPreprocessor

```python
class DataPreprocessor:
    def __init__(self):
        """Initialize preprocessor."""
    
    def clean_text(self, text: str) -> str:
        """Clean raw text."""
    
    def preprocess_text(self, text: str) -> str:
        """Complete preprocessing pipeline."""
    
    def preprocess_dataset(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
```

### FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self, max_features=3000, ngram_range=(1,2)):
        """Initialize feature extractor."""
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to features."""
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
    
    def get_feature_names(self) -> List[str]:
        """Get vocabulary."""
```

### NaiveBayesSpamClassifier

```python
class NaiveBayesSpamClassifier:
    def __init__(self, alpha=1.0):
        """Initialize classifier."""
    
    def train(self, X_train, y_train):
        """Train the model."""
    
    def predict(self, X_test) -> np.ndarray:
        """Predict class labels."""
    
    def predict_proba(self, X_test) -> np.ndarray:
        """Predict probabilities."""
    
    def save_model(self, filepath: str):
        """Save model to disk."""
    
    def load_model(self, filepath: str):
        """Load model from disk."""
```

### LogisticRegressionSpamClassifier

```python
class LogisticRegressionSpamClassifier:
    def __init__(self, C=1.0, max_iter=1000):
        """Initialize classifier."""
    
    def train(self, X_train, y_train):
        """Train the model."""
    
    def predict(self, X_test) -> np.ndarray:
        """Predict class labels."""
    
    def predict_proba(self, X_test) -> np.ndarray:
        """Predict probabilities."""
    
    def get_decision_function(self, X_test) -> np.ndarray:
        """Get raw decision values."""
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```
Error: ModuleNotFoundError: No module named 'sklearn'
```
Solution:
```bash
pip install scikit-learn
```

**2. NLTK Data Missing**
```
Error: Resource stopwords not found
```
Solution:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**3. Model Not Found**
```
Error: Model file not found
```
Solution:
```bash
# Run main.py first to train and save models
python main.py
```

**4. Port Already in Use (Web App)**
```
Error: Address already in use
```
Solution:
```python
# Change port in web_app.py
app.run(port=5001)  # Use different port
```

### Performance Issues

**Slow Training**
- Reduce max_features in FeatureExtractor
- Use smaller dataset for testing
- Enable parallel processing (already enabled in LR)

**Memory Issues**
- Reduce batch size
- Use sparse matrices (already implemented)
- Clear variables after use

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_spam_detector.py::TestNaiveBayesClassifier -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## Advanced Usage

### Custom Dataset

To use your own dataset:

```python
import pandas as pd
from src.data_preprocessing import DataPreprocessor

# Load your data
df = pd.DataFrame({
    'text': ['your', 'emails', 'here'],
    'label': [0, 1, 0]  # 0=ham, 1=spam
})

# Continue with normal pipeline
preprocessor = DataPreprocessor()
# ... rest of pipeline
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# For Naive Bayes
params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid = GridSearchCV(MultinomialNB(), params, cv=5)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_['alpha']

# For Logistic Regression
params = {'C': [0.1, 1.0, 10.0]}
grid = GridSearchCV(LogisticRegression(), params, cv=5)
grid.fit(X_train, y_train)
best_C = grid.best_params_['C']
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    classifier.model, 
    X, y, 
    cv=5, 
    scoring='f1'
)
print(f"CV F1 Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Contributing

### Guidelines

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit pull request

### Code Style

Follow PEP 8 guidelines:
```bash
# Install flake8
pip install flake8

# Check code
flake8 src/
```

---

## License

This project is for educational purposes.

---

## Acknowledgments

- SMS Spam Collection Dataset from UCI ML Repository
- scikit-learn documentation
- NLTK project

---

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review inline code comments
3. Run tests to verify setup
4. Check example notebooks

---

**Last Updated**: 2024
**Version**: 1.0.0
