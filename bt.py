from flask import Flask, request, render_template, session
import joblib
import numpy as np

bt = Flask(__name__)
bt.secret_key = 'your_secret_key'  # Đặt một khóa bí mật

# Tải các mô hình đã lưu
lr_model = joblib.load("linear_regression_model.pkl")
knn_model = joblib.load("knn_model.pkl")
svr_model = joblib.load("svr_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

@bt.route("/")
def home():
    # Lấy dữ liệu từ session nếu đã có (để điền lại form)
    tv = session.get('tv', '')
    radio = session.get('radio', '')
    newspaper = session.get('newspaper', '')
    model_choice = session.get('model_choice', 'linear_regression')
    return render_template("index.html", tv=tv, radio=radio, newspaper=newspaper, model_choice=model_choice)

@bt.route("/predict", methods=["POST"])
def predict():
    # Lấy dữ liệu từ form
    tv = float(request.form['tv'])
    radio = float(request.form['radio'])
    newspaper = float(request.form['newspaper'])
    model_choice = request.form["model"]
    
    # Lưu lại dữ liệu vào session để sử dụng lại
    session['tv'] = tv
    session['radio'] = radio
    session['newspaper'] = newspaper
    session['model_choice'] = model_choice

    # Chuẩn bị dữ liệu đầu vào cho mô hình
    final_features = np.array([[tv, radio, newspaper]])

    # Dự đoán từ mô hình đã chọn
    if model_choice == "linear_regression":
        prediction = lr_model.predict(final_features)
    elif model_choice == "knn":
        prediction = knn_model.predict(final_features)
    elif model_choice == "svr":
        prediction = svr_model.predict(final_features)
    elif model_choice == "random_forest":
        prediction = rf_model.predict(final_features)

    predicted_sales = prediction[0]
    return render_template("result.html", prediction=predicted_sales)

if __name__ == "__main__":
    bt.run(debug=True)
