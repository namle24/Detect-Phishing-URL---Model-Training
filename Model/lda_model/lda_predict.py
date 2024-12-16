import numpy as np
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

def load_model_and_tools(model_path, vectorizer_path, pca_path):
    lda_model = load(model_path)
    vectorizer = load(vectorizer_path)
    pca = load(pca_path)
    print("Mô hình và công cụ đã được tải thành công.")
    return lda_model, vectorizer, pca

def predict_url(url, lda_model, vectorizer, pca):
    url_tfidf = vectorizer.transform([url])
    url_pca = pca.transform(url_tfidf.toarray())
    prob = lda_model.predict_proba(url_pca)[:, 1][0]
    label = lda_model.predict(url_pca)[0]
    return label, prob


model_path = 'lda_model_crawl_data.joblib'
vectorizer_path = 'tfidf_vectorizer_crawl_data.joblib'
pca_path = 'pca_model_crawl_data.joblib'
lda_model, vectorizer, pca = load_model_and_tools(model_path, vectorizer_path, pca_path)


while True:
    print("\nNhập URL để phân loại (hoặc nhập 'exit' để thoát):")
    user_input = input("URL: ").strip()
    if user_input.lower() == 'exit':
        print("Đã thoát chương trình.")
        sys.exit()
    elif user_input:
        try:
            label, prob = predict_url(user_input, lda_model, vectorizer, pca)
            if label == 0:
                print(f"🔴 URL được phân loại là **Giả mạo** với xác suất {(1 - prob) * 100:.2f}%.")
            else:
                print(f"🟢 URL được phân loại là **An toàn** với xác suất {prob * 100:.2f}%.")
        except Exception as e:
            print(f"Đã xảy ra lỗi khi phân loại: {e}")
    else:
        print("Vui lòng nhập một URL hợp lệ.")
