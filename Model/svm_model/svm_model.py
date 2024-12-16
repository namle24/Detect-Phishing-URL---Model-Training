import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.feature_selection import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
import seaborn as sns
from imblearn.over_sampling import SMOTE
from joblib import dump
import time


def load_data(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    end_time = time.time()
    print(f"Time Data loading: {end_time - start_time:.2f} seconds.")
    return df


def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

df = load_data(r'D:\PyProject\pythonProject\Data\URL_dataset_crawl.csv')
print("Check Data:\n", df.isnull().sum())


X = df.drop(columns=['label'])
y = df['label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=42, stratify=y_temp)


print(f"Train size: {X_train.shape}")
print(f"Test size: {X_valid.shape}")
print(f"Kích thước tập test: {X_test.shape}")


start_time = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=5, max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train['URL'])
X_valid_tfidf = vectorizer.transform(X_valid['URL'])
X_test_tfidf = vectorizer.transform(X_test['URL'])
end_time = time.time()
print(f"Time Vectorization: {end_time - start_time:.2f} seconds.")


selector = SelectKBest(chi2, k=2000)
X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
X_valid_tfidf = selector.transform(X_valid_tfidf)
X_test_tfidf = selector.transform(X_test_tfidf)


start_time = time.time()
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
end_time = time.time()
print(f"Time SMOTE oversampling: {end_time - start_time:.2f} seconds.")


start_time = time.time()
svm_model = SVC(kernel='linear', C=0.1, probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")


save_model_and_vectorizer(svm_model, vectorizer, 'svm_model_crawl.joblib', 'tfidf_vectorizer_crawl.joblib')
y_pred_prob = svm_model.predict_proba(X_test_tfidf)[:, 1]
y_pred_binary = (y_pred_prob >= 0.6).astype(int)


accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)
print("=== SVM Metrics ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"F1 Score: {f1:.2f}")
report = classification_report(y_test, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'])
print("\nClassification Report:\n", report)


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='SVM')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve_svm.png')
plt.close()


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malicious (1)'], yticklabels=['Benign (0)', 'Malicious (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_svm.png')
plt.close()

print("Saved to 'roc_curve_svm.png' và 'confusion_matrix_svm.png'.")
