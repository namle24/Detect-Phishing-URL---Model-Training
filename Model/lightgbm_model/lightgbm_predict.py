import joblib
import lightgbm as lgb
import re


def load_model_and_vectorizer(model_path, vectorizer_path):
    try:
        model = lgb.Booster(model_file=model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Mô hình và vectorizer đã được tải thành công.")
        return model, vectorizer
    except FileNotFoundError:
        print("Không tìm thấy tệp mô hình hoặc vectorizer. Vui lòng kiểm tra lại đường dẫn.")
        exit()
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải mô hình hoặc vectorizer: {e}")
        exit()


def is_valid_url(url):
    regex = r'^(https?:\/\/)?([a-z0-9\-]+\.)+[a-z]{2,7}(:[0-9]{1,4})?(\/.*)?$'
    return bool(re.match(regex, url.strip()))


def detect_url(url, model, vectorizer):
    url_vectorized = vectorizer.transform([url])
    prob = model.predict(url_vectorized)[0]
    label = 1 if prob > 0.5 else 0
    return label, prob


if __name__ == "__main__":
    model_path = 'lightgbm_model_crawl_data.txt'
    vectorizer_path = 'tfidf_vectorizer_crawl_data.joblib'


    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)


    print("\n=== Phát hiện URL Phishing ===")
    while True:
        url = input("Nhập URL cần kiểm tra (hoặc nhập 'exit' để thoát): ").strip()

        if url.lower() == 'exit':
            print("Kết thúc chương trình. Hẹn gặp lại!")
            break

        if not url:
            print("URL không được để trống. Vui lòng nhập lại.")
            continue

        if not is_valid_url(url):
            print("URL không hợp lệ. Vui lòng nhập URL hợp lệ.")
            continue


        label, prob = detect_url(url, model, vectorizer)

        if label == 0:
            print(f"🔴 URL '{url}' được dự đoán là PHISHING (Xác suất: {1 - prob:.2f}).")
        else:
            print(f"🟢 URL '{url}' được dự đoán là AN TOÀN (Xác suất: {prob:.2f}).")
