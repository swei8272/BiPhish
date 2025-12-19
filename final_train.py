# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# import joblib
# import os
#
# try:
#     from sklearnex import patch_sklearn
#
#     patch_sklearn()
#     print("Using Intel scikit-learn extensions for acceleration")
# except ImportError:
#     print("Intel scikit-learn extension not found. Using standard scikit-learn.")
#
# print("=" * 80)
# print("PHISHING DETECTION - ENSEMBLE TRAINING PIPELINE")
# print("=" * 80)
#
# # Load Data
# print("\n[1/5] Loading data files...")
# try:
#     a = pd.read_csv('Legitimate_url_data_art.csv')
#     print(f"  ✓ Loaded Legitimate URL features (artificial): {a.shape}")
# except FileNotFoundError:
#     print("  ✗ Error: Legitimate_url_data_art.csv not found!")
#     print("  Run feature_extractor.py first to generate this file.")
#     exit(1)
#
# try:
#     b = pd.read_csv('Phishing_url_data_art.csv')
#     print(f"  ✓ Loaded Phishing URL features (artificial): {b.shape}")
# except FileNotFoundError:
#     print("  ✗ Error: Phishing_url_data_art.csv not found!")
#     print("  Run feature_extractor.py first to generate this file.")
#     exit(1)
#
# try:
#     c = pd.read_csv('Legitimate_url_data_cnn.csv')
#     print(f"  ✓ Loaded Legitimate URL features (CNN): {c.shape}")
# except FileNotFoundError:
#     print("  ✗ Error: Legitimate_url_data_cnn.csv not found!")
#     print("  Run CNN_process.py first to generate this file.")
#     exit(1)
#
# try:
#     d = pd.read_csv('Phishing_url_data_cnn.csv')
#     print(f"  ✓ Loaded Phishing URL features (CNN): {d.shape}")
# except FileNotFoundError:
#     print("  ✗ Error: Phishing_url_data_cnn.csv not found!")
#     print("  Run CNN_process.py first to generate this file.")
#     exit(1)
#
# # Add target labels
# print("\n[2/5] Adding target labels...")
# a['target'] = 0  # Legitimate
# b['target'] = 1  # Phishing
# c['target'] = 0  # Legitimate
# d['target'] = 1  # Phishing
#
# # Prepare Dataframes
# print("\n[3/5] Preparing feature matrices...")
#
# # Combine CNN features (with targets)
# cnndata = pd.concat([c, d], axis=0).reset_index(drop=True)
# print(f"  CNN features combined: {cnndata.shape}")
#
# # Combine artificial features (with targets)
# artdata_full = pd.concat([a, b], axis=0).reset_index(drop=True)
# print(f"  Artificial features combined: {artdata_full.shape}")
#
# # Select only the 10 features identified by RFE (from fdc.py analysis)
# selected_features = ['URL_length', 'URL_subdomains', 'URL_totalWordUrl',
#                      'URL_shortestWordPath', 'URL_longestWordUrl',
#                      'URL_longestWordHost', 'URL_longestWordPath',
#                      'URL_averageWordUrl', 'URL_averageWordHost',
#                      'URL_averageWordPath']
#
# # Check if all features exist
# missing_features = [f for f in selected_features if f not in artdata_full.columns]
# if missing_features:
#     print(f"  ✗ Warning: Missing features in artificial data: {missing_features}")
#     print(f"  Available columns: {list(artdata_full.columns)}")
#     # Use only available features
#     selected_features = [f for f in selected_features if f in artdata_full.columns]
#     print(f"  Using {len(selected_features)} available features")
#
# artdata_features = artdata_full[selected_features]
# print(f"  Selected artificial features: {artdata_features.shape}")
#
# # IMPORTANT: Extract target BEFORE merging to avoid confusion
# y = cnndata['target'].values
# print(f"  Target variable shape: {y.shape}")
#
# # Remove target from CNN data
# cnndata_features = cnndata.drop('target', axis=1)
# print(f"  CNN features (without target): {cnndata_features.shape}")
#
# # Concatenate features horizontally (axis=1)
# # Each row in cnndata_features should align with corresponding row in artdata_features
# alldata = pd.concat([cnndata_features, artdata_features], axis=1)
# print(f"  Combined feature matrix: {alldata.shape}")
#
# # Final feature matrix and target
# X = alldata.values
# print(f"  Final X shape: {X.shape}")
# print(f"  Final y shape: {y.shape}")
#
# # Check for NaN or Inf values
# if pd.DataFrame(X).isnull().any().any():
#     print("  ⚠ Warning: Found NaN values in features. Filling with 0...")
#     X = pd.DataFrame(X).fillna(0).values
#
# if pd.DataFrame(X).isin([np.inf, -np.inf]).any().any():
#     print("  ⚠ Warning: Found Inf values in features. Clipping...")
#     X = np.clip(X, -1e10, 1e10)
#
# # Split data
# print("\n[4/5] Splitting data (80% train, 20% test)...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(f"  Training set: {X_train.shape}, Test set: {X_test.shape}")
# print(f"  Training labels - Phishing: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")
# print(f"  Test labels - Phishing: {sum(y_test)}, Legitimate: {len(y_test) - sum(y_test)}")
#
# # Initialize Classifiers
# print("\n[5/5] Training classifiers...")
# classifiers = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Naive Bayes': GaussianNB(),
#     'SVM': SVC(probability=True, kernel='linear'),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
# }
#
# trained_classifiers = {}
#
# for name, clf in classifiers.items():
#     print(f"\n  Training {name}...")
#     try:
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)
#         print(f"    ✓ Accuracy: {score:.4f}")
#
#         # Save model
#         filename = f"{name.lower().replace(' ', '_')}_t.pkl"
#         joblib.dump(clf, filename)
#         print(f"    ✓ Saved to {filename}")
#
#         trained_classifiers[name.lower().replace(' ', '_')] = clf
#     except Exception as e:
#         print(f"    ✗ Error training {name}: {e}")
#
# # Voting Classifier
# print("\n" + "=" * 80)
# print("TRAINING VOTING CLASSIFIER (ENSEMBLE)")
# print("=" * 80)
#
# voting_classifier = VotingClassifier(
#     estimators=[
#         ('logreg', trained_classifiers['logistic_regression']),
#         ('naivebayes', trained_classifiers['naive_bayes']),
#         ('svm', trained_classifiers['svm']),
#         ('decisiontree', trained_classifiers['decision_tree']),
#         ('randomforest', trained_classifiers['random_forest'])
#     ],
#     voting='soft',  # Use probability-based voting
#     weights=[1, 1, 2, 1, 2]  # Higher weight for SVM and Random Forest
# )
#
# print("\nFitting ensemble model...")
# voting_classifier.fit(X_train, y_train)
# joblib.dump(voting_classifier, 'voting_classifier.pkl')
# print("✓ Saved voting_classifier.pkl")
#
# # Evaluate
# print("\n" + "=" * 80)
# print("FINAL EVALUATION")
# print("=" * 80)
#
# y_pred = voting_classifier.predict(X_test)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], digits=4))
#
# print("\n" + "=" * 80)
# print("TRAINING COMPLETE!")
# print("=" * 80)
# print("\nSaved models:")
# for name in trained_classifiers.keys():
#     print(f"  - {name}_t.pkl")
# print(f"  - voting_classifier.pkl (ENSEMBLE)")

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['KMP_WARNINGS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Using Intel scikit-learn extensions for acceleration")
except ImportError:
    print("Intel scikit-learn extension not found. Using standard scikit-learn.")

print("=" * 80)
print("PHISHING DETECTION - ENSEMBLE TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# [1/6] LOAD DATA FILES
# ============================================================================
print("\n[1/6] Loading data files...")

try:
    a = pd.read_csv('Legitimate_url_data_art.csv')
    print(f"  ✓ Loaded Legitimate URL features (artificial): {a.shape}")
except FileNotFoundError:
    print("  ✗ Error: Legitimate_url_data_art.csv not found!")
    print("  Run feature_extractor.py first to generate this file.")
    exit(1)

try:
    b = pd.read_csv('Phishing_url_data_art.csv')
    print(f"  ✓ Loaded Phishing URL features (artificial): {b.shape}")
except FileNotFoundError:
    print("  ✗ Error: Phishing_url_data_art.csv not found!")
    print("  Run feature_extractor.py first to generate this file.")
    exit(1)

try:
    c = pd.read_csv('Legitimate_url_data_cnn.csv')
    print(f"  ✓ Loaded Legitimate URL features (CNN): {c.shape}")
except FileNotFoundError:
    print("  ✗ Error: Legitimate_url_data_cnn.csv not found!")
    print("  Run CNN_process.py first to generate this file.")
    exit(1)

try:
    d = pd.read_csv('Phishing_url_data_cnn.csv')
    print(f"  ✓ Loaded Phishing URL features (CNN): {d.shape}")
except FileNotFoundError:
    print("  ✗ Error: Phishing_url_data_cnn.csv not found!")
    print("  Run CNN_process.py first to generate this file.")
    exit(1)

# ============================================================================
# [2/6] ADD TARGET LABELS
# ============================================================================
print("\n[2/6] Adding target labels...")
a['target'] = 0  # Legitimate
b['target'] = 1  # Phishing
c['target'] = 0  # Legitimate
d['target'] = 1  # Phishing

# ============================================================================
# [3/6] PREPARE FEATURE MATRICES
# ============================================================================
print("\n[3/6] Preparing feature matrices...")

# Combine CNN features (with targets)
cnndata = pd.concat([c, d], axis=0).reset_index(drop=True)
print(f"  CNN features combined: {cnndata.shape}")

# Combine artificial features (with targets)
artdata_full = pd.concat([a, b], axis=0).reset_index(drop=True)
print(f"  Artificial features combined: {artdata_full.shape}")

# Select only the 10 features identified by RFE (from fdc.py analysis)
selected_features = ['URL_length', 'URL_subdomains', 'URL_totalWordUrl',
                     'URL_shortestWordPath', 'URL_longestWordUrl',
                     'URL_longestWordHost', 'URL_longestWordPath',
                     'URL_averageWordUrl', 'URL_averageWordHost',
                     'URL_averageWordPath']

# Check if all features exist
missing_features = [f for f in selected_features if f not in artdata_full.columns]
if missing_features:
    print(f"  ⚠ Warning: Missing features in artificial data: {missing_features}")
    print(f"  Available columns: {list(artdata_full.columns)}")
    # Use only available features
    selected_features = [f for f in selected_features if f in artdata_full.columns]
    print(f"  Using {len(selected_features)} available features")

artdata_features = artdata_full[selected_features]
print(f"  Selected artificial features: {artdata_features.shape}")

# CRITICAL: Extract target BEFORE merging to avoid duplication
y = cnndata['target'].values
print(f"  Target variable shape: {y.shape}")

# Remove target from CNN data
cnndata_features = cnndata.drop('target', axis=1)
print(f"  CNN features (without target): {cnndata_features.shape}")

# Concatenate features horizontally (axis=1)
alldata = pd.concat([cnndata_features, artdata_features], axis=1)
print(f"  Combined feature matrix: {alldata.shape}")

# Final feature matrix and target
X = alldata.values
print(f"  Final X shape: {X.shape}")
print(f"  Final y shape: {y.shape}")

# Check for NaN or Inf values
nan_count = pd.DataFrame(X).isnull().sum().sum()
if nan_count > 0:
    print(f"  ⚠ Warning: Found {nan_count} NaN values. Filling with 0...")
    X = pd.DataFrame(X).fillna(0).values

inf_count = pd.DataFrame(X).isin([np.inf, -np.inf]).sum().sum()
if inf_count > 0:
    print(f"  ⚠ Warning: Found {inf_count} Inf values. Clipping...")
    X = np.clip(X, -1e10, 1e10)

# ============================================================================
# [4/6] SPLIT DATA
# ============================================================================
print("\n[4/6] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"  Training labels - Phishing: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")
print(f"  Test labels - Phishing: {sum(y_test)}, Legitimate: {len(y_test) - sum(y_test)}")

# ============================================================================
# [5/6] SCALE FEATURES (Critical for Logistic Regression convergence)
# ============================================================================
print("\n[5/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print("  ✓ Features scaled and scaler saved to scaler.pkl")
print("  ⚠ IMPORTANT: Use this scaler for inference!")

# ============================================================================
# [6/6] TRAIN CLASSIFIERS
# ============================================================================
print("\n[6/6] Training classifiers...")
print("=" * 80)

trained_classifiers = {}

# 1. Logistic Regression (optimized for large datasets)
print("\n  [1/5] Training Logistic Regression...")
try:
    logistic_regressor = LogisticRegression(
        max_iter=2000,
        solver='saga',      # Fast for large datasets
        n_jobs=-1,          # Use all CPU cores
        random_state=42
    )
    logistic_regressor.fit(X_train_scaled, y_train)
    score = logistic_regressor.score(X_test_scaled, y_test)
    print(f"        ✓ Accuracy: {score:.4f}")
    joblib.dump(logistic_regressor, 'logistic_regression_t.pkl')
    print(f"        ✓ Saved to logistic_regression_t.pkl")
    trained_classifiers['logistic_regression'] = logistic_regressor
except Exception as e:
    print(f"        ✗ Error: {e}")

# 2. Naive Bayes
print("\n  [2/5] Training Naive Bayes...")
try:
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train_scaled, y_train)
    score = naive_bayes.score(X_test_scaled, y_test)
    print(f"        ✓ Accuracy: {score:.4f}")
    joblib.dump(naive_bayes, 'naive_bayes_t.pkl')
    print(f"        ✓ Saved to naive_bayes_t.pkl")
    trained_classifiers['naive_bayes'] = naive_bayes
except Exception as e:
    print(f"        ✗ Error: {e}")

# 3. Linear SVM (FAST VERSION - 100x faster than kernel SVM)
print("\n  [3/5] Training Linear SVM (optimized for large datasets)...")
print("        (Using LinearSVC instead of SVC for speed)")
try:
    svm_base = LinearSVC(
        max_iter=2000,
        dual=False,         # Faster when n_samples > n_features
        random_state=42
    )
    # Wrap with CalibratedClassifierCV to get probability estimates
    svm_classifier = CalibratedClassifierCV(svm_base, cv=3)
    svm_classifier.fit(X_train_scaled, y_train)
    score = svm_classifier.score(X_test_scaled, y_test)
    print(f"        ✓ Accuracy: {score:.4f}")
    print(f"        ✓ Training time: ~5-10 min (vs 2+ hours with kernel SVM)")
    joblib.dump(svm_classifier, 'svm_classifier_t.pkl')
    print(f"        ✓ Saved to svm_classifier_t.pkl")
    trained_classifiers['svm'] = svm_classifier
except Exception as e:
    print(f"        ✗ Error: {e}")

# 4. Decision Tree
print("\n  [4/5] Training Decision Tree...")
try:
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train_scaled, y_train)
    score = decision_tree.score(X_test_scaled, y_test)
    print(f"        ✓ Accuracy: {score:.4f}")
    joblib.dump(decision_tree, 'decision_tree_t.pkl')
    print(f"        ✓ Saved to decision_tree_t.pkl")
    trained_classifiers['decision_tree'] = decision_tree
except Exception as e:
    print(f"        ✗ Error: {e}")

# 5. Random Forest
print("\n  [5/5] Training Random Forest...")
try:
    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1           # Use all CPU cores
    )
    random_forest.fit(X_train_scaled, y_train)
    score = random_forest.score(X_test_scaled, y_test)
    print(f"        ✓ Accuracy: {score:.4f}")
    joblib.dump(random_forest, 'random_forest_t.pkl')
    print(f"        ✓ Saved to random_forest_t.pkl")
    trained_classifiers['random_forest'] = random_forest
except Exception as e:
    print(f"        ✗ Error: {e}")

# ============================================================================
# CREATE VOTING CLASSIFIER (ENSEMBLE)
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VOTING CLASSIFIER (ENSEMBLE)")
print("=" * 80)

if len(trained_classifiers) < 5:
    print(f"\n⚠ Warning: Only {len(trained_classifiers)}/5 classifiers trained successfully.")
    print("Ensemble will use available classifiers only.")

voting_classifier = VotingClassifier(
    estimators=[
        ('logreg', trained_classifiers['logistic_regression']),
        ('naivebayes', trained_classifiers['naive_bayes']),
        ('svm', trained_classifiers['svm']),
        ('decisiontree', trained_classifiers['decision_tree']),
        ('randomforest', trained_classifiers['random_forest'])
    ],
    voting='soft',          # Use probability-based voting
    weights=[1, 1, 2, 1, 2] # Higher weight for SVM and Random Forest
)

print("\nFitting ensemble model...")
voting_classifier.fit(X_train_scaled, y_train)
joblib.dump(voting_classifier, 'voting_classifier.pkl')
print("✓ Saved to voting_classifier.pkl")

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

y_pred = voting_classifier.predict(X_test_scaled)
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'], digits=4))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

print("\nSaved models:")
for name in trained_classifiers.keys():
    print(f"  ✓ {name}_t.pkl")
print(f"  ✓ voting_classifier.pkl (ENSEMBLE)")
print(f"  ✓ scaler.pkl (IMPORTANT FOR INFERENCE)")

print("\n" + "=" * 80)
print("USAGE FOR INFERENCE:")
print("=" * 80)
print("""
import joblib
import numpy as np

# Load models
scaler = joblib.load('scaler.pkl')
model = joblib.load('voting_classifier.pkl')

# Prepare features (138 features: 128 CNN + 10 URL)
X_new = np.array([[...]])  # Your feature vector

# CRITICAL: Scale features before prediction
X_new_scaled = scaler.transform(X_new)

# Predict
prediction = model.predict(X_new_scaled)
probability = model.predict_proba(X_new_scaled)

print(f"Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
print(f"Confidence: {max(probability[0]):.2%}")
""")
print("=" * 80)