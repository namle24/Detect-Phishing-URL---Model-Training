import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import time


def load_data(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    end_time = time.time()
    print(f"Time Data loading: {end_time - start_time:.2f} seconds.")
    return df


def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")


df = load_data(r'D:\PyProject\pythonProject\Data\URL_dataset_no_crawl.csv')
print("Chẹck Data:\n", df.isnull().sum())
X = df.drop(columns=['label'])
y = df['label']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=42, stratify=y_temp)


print(f"Train Size: {X_train.shape}")
print(f"Valid Size: {X_valid.shape}")
print(f"Test Size: {X_test.shape}")


start_time = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['URL'])
X_valid_tfidf = vectorizer.transform(X_valid['URL'])
X_test_tfidf = vectorizer.transform(X_test['URL'])
end_time = time.time()
print(f"Time Vectorization: {end_time - start_time:.2f} seconds.")


start_time = time.time()
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
end_time = time.time()
print(f"Time SMOTE oversampling: {end_time - start_time:.2f} seconds.")


train_data = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
valid_data = xgb.DMatrix(X_valid_tfidf, label=y_valid)
test_data = xgb.DMatrix(X_test_tfidf, label=y_test)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.001,
    'max_depth': 5,
    'min_child_weight': 20,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 2,
    'alpha': 2,
    'seed': 42,
}


start_time = time.time()
evals = [(train_data, 'train'), (valid_data, 'eval')]
model = xgb.train(params, train_data, num_boost_round=10000, evals=evals, early_stopping_rounds=50)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")


save_model_and_vectorizer(model, vectorizer, 'xgboost_model_no_crawl.joblib', 'tfidf_vectorizer_no_crawl.joblib')


y_pred_prob = model.predict(test_data)
y_pred_binary = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)

print("\n=== XGB Metrics ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"F1 Score: {f1:.2f}")


report = classification_report(y_test, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'])
print("\nClassification Report:\n", report)


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='XGBoost')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve_xgboost.png')
plt.close()


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malicious (1)'], yticklabels=['Benign (0)', 'Malicious (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_xgboost.png')
plt.close()

print("Saved to 'roc_curve_xgboost.png' và 'confusion_matrix_xgboost.png'.")
