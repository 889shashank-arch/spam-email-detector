"""
Main Execution Script for Spam Email Detector

This script runs the complete spam detection pipeline:
1. Load and preprocess data
2. Extract features
3. Train Naive Bayes classifier
4. Train Logistic Regression classifier
5. Evaluate both models
6. Compare performance
7. Save models and results
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, load_data, split_data
from feature_extraction import FeatureExtractor
from naive_bayes_classifier import NaiveBayesSpamClassifier
from logistic_regression_classifier import LogisticRegressionSpamClassifier
from model_evaluation import ModelEvaluator, evaluate_model


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Execute the complete spam detection pipeline."""
    
    print_header("SPAM EMAIL DETECTOR - MACHINE LEARNING PROJECT")
    print("\nThis project demonstrates email spam classification using:")
    print("  • Naive Bayes Algorithm")
    print("  • Logistic Regression Algorithm")
    print("  • TF-IDF Feature Extraction")
    print("  • Comprehensive Model Evaluation\n")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ========== STEP 1: LOAD DATA ==========
    print_header("STEP 1: LOADING DATA")
    
    print("\nLoading dataset...")
    df = load_data()
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(df)}")
    print(f"  Spam messages: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
    print(f"  Ham messages: {len(df) - df['label'].sum()} ({(len(df) - df['label'].sum())/len(df)*100:.1f}%)")
    
    # Show sample data
    print("\nSample emails:")
    for i in range(min(3, len(df))):
        label = "SPAM" if df.iloc[i]['label'] == 1 else "HAM"
        text = df.iloc[i]['text'][:80] + "..." if len(df.iloc[i]['text']) > 80 else df.iloc[i]['text']
        print(f"  [{label}] {text}")
    
    # ========== STEP 2: PREPROCESS DATA ==========
    print_header("STEP 2: PREPROCESSING DATA")
    
    print("\nPreprocessing pipeline:")
    print("  • Converting to lowercase")
    print("  • Removing URLs and email addresses")
    print("  • Removing special characters and numbers")
    print("  • Tokenization")
    print("  • Removing stopwords")
    print("  • Lemmatization")
    
    preprocessor = DataPreprocessor()
    df['processed_text'] = preprocessor.preprocess_dataset(df['text'].tolist())
    
    print("\n✓ Text preprocessing completed")
    
    # Show example
    print("\nExample preprocessing:")
    sample_idx = 0
    print(f"  Original:   {df.iloc[sample_idx]['text'][:100]}...")
    print(f"  Processed:  {df.iloc[sample_idx]['processed_text'][:100]}...")
    
    # ========== STEP 3: FEATURE EXTRACTION ==========
    print_header("STEP 3: EXTRACTING FEATURES")
    
    print("\nExtracting TF-IDF features...")
    print("  • Max features: 3000")
    print("  • N-gram range: (1, 2) - unigrams and bigrams")
    print("  • Min document frequency: 2")
    print("  • Max document frequency: 0.95")
    
    extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 2))
    X = extractor.fit_transform(df['processed_text'].tolist())
    y = df['label'].values
    
    print(f"\n✓ Feature extraction completed")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Total vocabulary size: {len(extractor.get_feature_names())}")
    
    # Show top features by class
    top_ham, top_spam = extractor.get_top_features_by_class(X, y, top_n=10)
    print("\n  Top Ham indicators:")
    for i, feature in enumerate(top_ham[:5], 1):
        print(f"    {i}. {feature}")
    
    print("\n  Top Spam indicators:")
    for i, feature in enumerate(top_spam[:5], 1):
        print(f"    {i}. {feature}")
    
    # ========== STEP 4: SPLIT DATA ==========
    print_header("STEP 4: SPLITTING DATA")
    
    print("\nSplitting into train and test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    print(f"✓ Data split completed")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Training spam ratio: {y_train.sum()/len(y_train)*100:.1f}%")
    print(f"  Test spam ratio: {y_test.sum()/len(y_test)*100:.1f}%")
    
    # ========== STEP 5: TRAIN NAIVE BAYES ==========
    print_header("STEP 5: TRAINING NAIVE BAYES CLASSIFIER")
    
    print("\nInitializing Multinomial Naive Bayes...")
    print("  • Algorithm: Multinomial Naive Bayes")
    print("  • Smoothing: Laplace (alpha=1.0)")
    
    nb_classifier = NaiveBayesSpamClassifier(alpha=1.0)
    nb_classifier.train(X_train, y_train)
    
    print("\n✓ Naive Bayes training completed")
    
    # Show class priors
    priors = nb_classifier.get_class_priors()
    print(f"  Class priors learned:")
    print(f"    P(Ham) = {priors[0]:.4f}")
    print(f"    P(Spam) = {priors[1]:.4f}")
    
    # ========== STEP 6: TRAIN LOGISTIC REGRESSION ==========
    print_header("STEP 6: TRAINING LOGISTIC REGRESSION CLASSIFIER")
    
    print("\nInitializing Logistic Regression...")
    print("  • Algorithm: Logistic Regression")
    print("  • Regularization: L2 (C=1.0)")
    print("  • Solver: L-BFGS")
    print("  • Class weighting: Balanced")
    
    lr_classifier = LogisticRegressionSpamClassifier(C=1.0, max_iter=1000)
    lr_classifier.train(X_train, y_train)
    
    print("\n✓ Logistic Regression training completed")
    print(f"  Learned bias term: {lr_classifier.bias:.4f}")
    print(f"  Number of features with weights: {len(lr_classifier.weights)}")
    
    # ========== STEP 7: EVALUATE MODELS ==========
    print_header("STEP 7: EVALUATING MODELS")
    
    print("\nEvaluating Naive Bayes classifier...")
    nb_metrics = evaluate_model(nb_classifier, X_test, y_test, 
                                "Naive Bayes", save_dir='results')
    
    print("\nEvaluating Logistic Regression classifier...")
    lr_metrics = evaluate_model(lr_classifier, X_test, y_test,
                               "Logistic Regression", save_dir='results')
    
    # ========== STEP 8: COMPARE MODELS ==========
    print_header("STEP 8: COMPARING MODELS")
    
    evaluator = ModelEvaluator()
    
    # Comparison table
    print("\n" + " "*20 + "PERFORMANCE COMPARISON")
    print("-"*70)
    print(f"{'Metric':<20} {'Naive Bayes':>15} {'Logistic Regression':>20}")
    print("-"*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        if metric in nb_metrics and metric in lr_metrics:
            nb_val = nb_metrics[metric]
            lr_val = lr_metrics[metric]
            winner = "🏆" if nb_val > lr_val else ("🏆" if lr_val > nb_val else "")
            print(f"{metric.capitalize():<20} {nb_val:>15.4f} {lr_val:>15.4f}  {winner}")
    
    print("-"*70)
    
    # Generate comparison plots
    print("\nGenerating comparison visualizations...")
    
    results = {
        'Naive Bayes': nb_metrics,
        'Logistic Regression': lr_metrics
    }
    evaluator.compare_models(results, save_path='results/model_comparison.png')
    
    # Compare ROC curves
    nb_proba = nb_classifier.predict_proba(X_test)[:, 1]
    lr_proba = lr_classifier.predict_proba(X_test)[:, 1]
    
    models_data = [
        ('Naive Bayes', y_test, nb_proba),
        ('Logistic Regression', y_test, lr_proba)
    ]
    evaluator.compare_roc_curves(models_data, save_path='results/roc_comparison.png')
    
    # ========== STEP 9: SAVE MODELS ==========
    print_header("STEP 9: SAVING MODELS")
    
    print("\nSaving trained models...")
    nb_classifier.save_model('models/naive_bayes_model.pkl')
    lr_classifier.save_model('models/logistic_regression_model.pkl')
    
    print("\n✓ Models saved successfully")
    
    # ========== STEP 10: TEST PREDICTIONS ==========
    print_header("STEP 10: TESTING PREDICTIONS")
    
    test_emails = [
        "Congratulations! You have won a $1000 gift card. Click here to claim now!",
        "Hi, can we schedule a meeting for tomorrow at 2 PM?",
        "URGENT: Your account has been compromised. Verify immediately!",
        "Thanks for the report. I'll review it and get back to you."
    ]
    
    print("\nTesting on new emails:\n")
    
    # Preprocess test emails
    processed_test = preprocessor.preprocess_dataset(test_emails)
    test_features = extractor.transform(processed_test)
    
    # Get predictions from both models
    nb_predictions = nb_classifier.predict(test_features)
    nb_probas = nb_classifier.predict_proba(test_features)
    
    lr_predictions = lr_classifier.predict(test_features)
    lr_probas = lr_classifier.predict_proba(test_features)
    
    for i, email in enumerate(test_emails):
        print(f"Email: {email}")
        
        nb_label = "SPAM" if nb_predictions[i] == 1 else "HAM"
        nb_conf = nb_probas[i][nb_predictions[i]] * 100
        
        lr_label = "SPAM" if lr_predictions[i] == 1 else "HAM"
        lr_conf = lr_probas[i][lr_predictions[i]] * 100
        
        print(f"  Naive Bayes:        {nb_label} (confidence: {nb_conf:.1f}%)")
        print(f"  Logistic Regression: {lr_label} (confidence: {lr_conf:.1f}%)")
        print()
    
    # ========== FINAL SUMMARY ==========
    print_header("PROJECT SUMMARY")
    
    print("\n✅ Project completed successfully!\n")
    print("Generated files:")
    print("  📁 models/")
    print("     • naive_bayes_model.pkl")
    print("     • logistic_regression_model.pkl")
    print("  📁 results/")
    print("     • naive_bayes_confusion_matrix.png")
    print("     • logistic_regression_confusion_matrix.png")
    print("     • naive_bayes_roc_curve.png")
    print("     • logistic_regression_roc_curve.png")
    print("     • model_comparison.png")
    print("     • roc_comparison.png")
    
    print("\n" + "="*70)
    print("  Thank you for using the Spam Email Detector!")
    print("="*70 + "\n")
    
    # Determine best model
    if nb_metrics['f1_score'] > lr_metrics['f1_score']:
        best_model = "Naive Bayes"
        best_f1 = nb_metrics['f1_score']
    else:
        best_model = "Logistic Regression"
        best_f1 = lr_metrics['f1_score']
    
    print(f"🏆 Best performing model: {best_model} (F1-Score: {best_f1:.4f})")
    print()

        # evaluation code above finishes here

    print("\n" + "="*60)
    print(" MANUAL EMAIL SPAM TEST ")
    print("="*60)

    while True:
        user_email = input("\nEnter an email message (or type 'exit'): ")

        if user_email.lower() == 'exit':
            print("Exiting prediction...")
            break

        processed = preprocessor.preprocess_text(user_email)
        vector = extractor.transform([processed])


        nb_pred = nb_classifier.predict(vector)[0]
        lr_pred = lr_classifier.predict(vector)[0]

        print("\nPrediction Results:")
        print("Naive Bayes:", "SPAM 🚨" if nb_pred == 1 else "HAM ✅")
        print("Logistic Regression:", "SPAM 🚨" if lr_pred == 1 else "HAM ✅")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
