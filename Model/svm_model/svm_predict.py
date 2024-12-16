import joblib
import numpy as np

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def detect_url(url, model, vectorizer):
    url_vectorized = vectorizer.transform([url])
    prob = model.predict_proba(url_vectorized)[0][1]
    label = model.predict(url_vectorized)[0]
    return label, prob

if __name__ == "__main__":
    model_path = 'svm_model_no_crawl.joblib'
    vectorizer_path = 'tfidf_vectorizer_no_crawl.joblib'
    try:
        model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        print("Mô hình và vectorizer đã được tải thành công.")
    except FileNotFoundError:
        print("Không tìm thấy tệp mô hình hoặc vectorizer. Vui lòng kiểm tra lại.")
        exit()

    print("\n=== Phát hiện URL Phishing ===")
    while True:
        url = input("Nhập URL cần kiểm tra (hoặc nhập 'exit' để thoát): ").strip()
        if url.lower() == 'exit':
            print("Kết thúc chương trình. Hẹn gặp lại!")
            break
        elif url == "":
            print("URL không được để trống. Vui lòng nhập lại.")
            continue

        label, prob = detect_url(url, model, vectorizer)
        if label == 0:
            print(f"🔴 URL '{url}' được dự đoán là PHISHING (Xác suất: {1-prob:.2f}).")
        else:
            print(f"🟢 URL '{url}' được dự đoán là AN TOÀN (Xác suất: {prob:.2f}).")
