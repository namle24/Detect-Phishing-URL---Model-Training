import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from joblib import dump
import time


def load_data(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    end_time = time.time()
    print(f"Time Data loading completed in {end_time - start_time:.2f} seconds.")
    return df


def save_model_and_tools(model, vectorizer, pca, model_path, vectorizer_path, pca_path):
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    dump(pca, pca_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print(f"PCA saved to {pca_path}")


df = load_data(r'D:\PyProject\pythonProject\Data\URL_dataset_crawl.csv')


print("Check Data:\n", df.isnull().sum())


X = df.drop(columns=['label'])
y = df['label']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=42, stratify=y_temp)

print(f"Train Size: {X_train.shape}")
print(f"Valid Size: {X_valid.shape}")
print(f"Test size: {X_test.shape}")


start_time = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
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


start_time = time.time()
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_resampled.toarray())
X_valid_pca = pca.transform(X_valid_tfidf.toarray())
X_test_pca = pca.transform(X_test_tfidf.toarray())
end_time = time.time()
print(f"Time PCA dimensionality reduction: {end_time - start_time:.2f} seconds.")


lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
start_time = time.time()
lda_model.fit(X_train_pca, y_train_resampled)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")


save_model_and_tools(lda_model, vectorizer, pca, 'lda_model_crawl_data.joblib', 'tfidf_vectorizer_crawl_data.joblib',
                     'pca_model_no_crawl_data.joblib')


y_pred_prob = lda_model.predict_proba(X_test_pca)[:, 1]
y_pred_binary = lda_model.predict(X_test_pca)


accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"F1 Score: {f1:.2f}")


report = classification_report(y_test, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'])
print("\nClassification Report:\n", report)


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='LDA')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve_lda.png')
plt.close()


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malicious (1)'], yticklabels=['Benign (0)', 'Malicious (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_lda.png')
plt.close()

print("Saced to 'roc_curve_lda.png' và 'confusion_matrix_lda.png'.")
