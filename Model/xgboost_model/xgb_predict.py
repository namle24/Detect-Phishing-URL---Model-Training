import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model và vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    vectorizer = joblib.load(vectorizer_path)
    print(f"Vectorizer loaded from {vectorizer_path}")
    return model, vectorizer

# Dự đoán URL
def predict_url(url, model, vectorizer):
    # Vectorize URL
    url_tfidf = vectorizer.transform([url])
    # Convert to DMatrix for XGBoost
    url_dmatrix = xgb.DMatrix(url_tfidf)
    # Predict probability
    pred_prob = model.predict(url_dmatrix)
    # Convert to binary prediction
    pred_label = 1 if pred_prob > 0.5 else 0
    return pred_label, pred_prob

# Đường dẫn tới model và vectorizer đã lưu
model_path = 'xgboost_model_no_crawl.joblib'
vectorizer_path = 'tfidf_vectorizer_no_crawl.joblib'

# Load model và vectorizer
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

# Nhập URL từ người dùng
while True:
    user_url = input("\nNhập URL để kiểm tra (hoặc nhập 'exit' để thoát): ").strip()
    if user_url.lower() == 'exit':
        print("Thoát chương trình.")
        break

    # Dự đoán URL
    label, prob = predict_url(user_url, model, vectorizer)
    if label == 0:
        print(f"🔴 URL '{user_url}' NGUY HIỂM với xác suất {prob[0]:.2f}")
    else:
        print(f"🟢 URL '{user_url}' AN TOÀN với xác suất {1 - prob[0]:.2f}")
