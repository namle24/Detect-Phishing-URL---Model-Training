import pandas as pd
from sklearn.feature_extraction.text import *
from lightgbm import Dataset as lgbDataset, train as lgb_train, early_stopping
from sklearn.metrics import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib
import time


def load_data(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    end_time = time.time()
    print(f"Data loading successful in {end_time - start_time:.2f} seconds.")
    return df


def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")


df = load_data(r'D:\PyProject\pythonProject\Data\URL_dataset_crawl.csv')


print("Check Data:\n", df.isnull().sum())


X = df.drop(columns=['label'])
y = df['label']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)


print(f"Train size: {X_train.shape}")
print(f"Valid size: {X_valid.shape}")
print(f"Test size: {X_test.shape}")


start_time = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['URL'])
X_valid_tfidf = vectorizer.transform(X_valid['URL'])
X_test_tfidf = vectorizer.transform(X_test['URL'])
end_time = time.time()
print(f"TIme Vectorization: {end_time - start_time:.2f} seconds.")


start_time = time.time()
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
end_time = time.time()
print(f"Time SMOTE oversampling: {end_time - start_time:.2f} seconds.")


train_data = lgbDataset(X_train_resampled, label=y_train_resampled)
valid_data = lgbDataset(X_valid_tfidf, label=y_valid, reference=train_data)


params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'learning_rate': 0.001,
    'max_depth': -1,
    'num_leaves': 8,
    'min_child_samples': 50,
    'min_data_in_leaf': 50,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}


start_time = time.time()
model = lgb_train(
    params,
    train_data,
    num_boost_round=10000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'eval'],
    callbacks=[early_stopping(stopping_rounds=50)]
)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

save_model_and_vectorizer(model, vectorizer, 'lightgbm_model_crawl_data.txt', 'tfidf_vectorizer_crawl_data.joblib')

y_pred_prob = model.predict(X_test_tfidf)
y_pred_binary = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred_binary)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"F1 score: {f1:.2f}")

report = classification_report(y_test, y_pred_binary, target_names=['Benign (0)', 'Malicious (1)'])
print("\nClassification Report:\n", report)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='LightGBM')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve_lightgbm.png')
plt.close()


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malicious (1)'], yticklabels=['Benign (0)', 'Malicious (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_lightgbm.png')
plt.close()

print("Saved in 'roc_curve_lightgbm.png' và 'confusion_matrix_lightgbm.png'.")
