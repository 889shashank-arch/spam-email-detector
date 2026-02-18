# Spam Email Detector

A comprehensive machine learning project for classifying emails as spam or ham (legitimate emails) using Naive Bayes and Logistic Regression algorithms.

## 📋 Project Overview

This project implements a spam email detection system with:
- **Two classification algorithms**: Naive Bayes and Logistic Regression
- **Comprehensive data preprocessing** and feature extraction
- **Model evaluation** with multiple metrics
- **Visualization** of results and performance
- **Web interface** for real-time email classification
- **Complete documentation** and testing

## 🎯 Features

- ✅ Data preprocessing and cleaning
- ✅ TF-IDF feature extraction
- ✅ Naive Bayes classifier implementation
- ✅ Logistic Regression classifier implementation
- ✅ Model comparison and evaluation
- ✅ Performance visualization
- ✅ Confusion matrices and ROC curves
- ✅ Interactive web interface
- ✅ Model persistence (save/load)
- ✅ Comprehensive testing

## 📂 Project Structure

```
spam_email_detector/
├── data/                   # Dataset files
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── naive_bayes_classifier.py
│   ├── logistic_regression_classifier.py
│   ├── model_evaluation.py
│   └── web_app.py
├── tests/                  # Unit tests
├── results/               # Output visualizations and reports
├── requirements.txt       # Python dependencies
├── main.py               # Main execution script
└── README.md             # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project:
```bash
cd spam_email_detector
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (run once):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 📊 Dataset

The project uses the SMS Spam Collection Dataset, which contains:
- **5,574 messages** (SMS/Email texts)
- **747 spam messages** (13.4%)
- **4,827 ham messages** (86.6%)

The dataset is automatically downloaded when you run the project.

## 🎮 Usage

### Option 1: Run Complete Pipeline

Run the main script to train models, evaluate, and generate visualizations:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Extract features using TF-IDF
3. Train both Naive Bayes and Logistic Regression models
4. Evaluate models and display metrics
5. Generate visualization plots
6. Save trained models

### Option 2: Interactive Web Interface

Launch the web application for real-time email classification:

```bash
python src/web_app.py
```

Then open your browser to `http://localhost:5000`

### Option 3: Jupyter Notebook

Explore the analysis interactively:

```bash
jupyter notebook notebooks/spam_detection_analysis.ipynb
```

### Option 4: Use as Python Module

```python
from src.naive_bayes_classifier import NaiveBayesSpamClassifier
from src.logistic_regression_classifier import LogisticRegressionSpamClassifier

# Initialize classifier
nb_classifier = NaiveBayesSpamClassifier()

# Train
nb_classifier.train(X_train, y_train)

# Predict
predictions = nb_classifier.predict(X_test)
```

## 📈 Model Performance

### Naive Bayes Classifier
- **Accuracy**: ~97.5%
- **Precision**: ~96.8%
- **Recall**: ~94.2%
- **F1-Score**: ~95.5%

### Logistic Regression Classifier
- **Accuracy**: ~98.1%
- **Precision**: ~97.5%
- **Recall**: ~95.8%
- **F1-Score**: ~96.6%

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/ -v
```

## 📊 Visualizations

The project generates several visualizations in the `results/` directory:
- Confusion matrices for both models
- ROC curves comparison
- Precision-Recall curves
- Feature importance plots
- Performance comparison bar charts

## 🔧 Key Components

### 1. Data Preprocessing (`data_preprocessing.py`)
- Text cleaning (remove special characters, URLs, numbers)
- Tokenization
- Stopword removal
- Lemmatization
- Lowercasing

### 2. Feature Extraction (`feature_extraction.py`)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-gram features (unigrams and bigrams)
- Configurable vocabulary size

### 3. Naive Bayes Classifier (`naive_bayes_classifier.py`)
- Multinomial Naive Bayes implementation
- Laplace smoothing
- Probability calculations

### 4. Logistic Regression Classifier (`logistic_regression_classifier.py`)
- L2 regularization
- Gradient descent optimization
- Probability predictions

### 5. Model Evaluation (`model_evaluation.py`)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Cross-validation
- Statistical significance testing

## 🎓 Educational Value

This project demonstrates:
- **Machine Learning Pipeline**: From data loading to model deployment
- **Algorithm Implementation**: Understanding Naive Bayes and Logistic Regression
- **NLP Techniques**: Text preprocessing and feature extraction
- **Model Evaluation**: Comprehensive metrics and visualization
- **Software Engineering**: Modular code, testing, documentation
- **Web Development**: Flask-based interactive interface

## 📝 Documentation

Each module contains detailed docstrings explaining:
- Function parameters and return values
- Algorithm explanations
- Usage examples
- Implementation details

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new features
- Improve algorithms
- Enhance visualizations
- Add more classifiers

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI Machine Learning Repository
- scikit-learn for reference implementations
- NLTK for NLP tools

## 📞 Support

For questions or issues, refer to the inline documentation or comments in the code.

---

**Created for academic demonstration of spam email classification using machine learning.**
