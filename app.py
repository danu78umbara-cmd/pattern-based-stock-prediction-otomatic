from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import preprocessor as pp
import joblib
import os
import io
import yfinance as yf

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")


def feature_engineering(data_raw):
    return pp.process_dataframe(data_raw)


def load_best_model(candle_pattern):
    folder = "saved_models"
    safe_pattern = candle_pattern.replace(" ", "_")

    for file in os.listdir(folder):
        if candle_pattern in file or safe_pattern in file:
            return joblib.load(os.path.join(folder, file))

    raise FileNotFoundError(
        f"Model untuk pola '{candle_pattern}' tidak ditemukan."
    )


def fetch_yahoo_data(ticker):
    try:
        # DI SINI TIDAK BOLEH ADA "+ '.JK' " LAGI!!
        df = yf.download(ticker, period="30d", interval="1d", threads=True)

        if df.empty:
            raise ValueError("Data Yahoo Finance kosong atau ticker tidak valid.")

        # Flatten multiindex (jika muncul)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })

        return df

    except Exception as e:
        raise ValueError(f"Error mengambil data Yahoo Finance: {str(e)}")

@app.route("/predict_ticker", methods=["POST"])
def predict_ticker():
    try:
        data = request.get_json()

        if not data or "ticker" not in data:
            return jsonify({
                "status": "error",
                "message": "Parameter 'ticker' wajib ada."
            }), 400

        ticker = data["ticker"].upper().strip()

        # Auto-add .JK kalau belum ada
        if not ticker.endswith(".JK"):
            ticker = ticker + ".JK"

        # --- Ambil data dari Yahoo Finance ---
        df_raw = fetch_yahoo_data(ticker)


        if len(df_raw) < 20:
            return jsonify({"status": "error", "message": "Data terlalu sedikit."}), 400

        df_feat = feature_engineering(df_raw)

        if df_feat.empty:
            return jsonify({"status": "error", "message": "Preprocessing menghasilkan data kosong."}), 400

        if "CandlePattern" not in df_feat.columns:
            return jsonify({
                "status": "error",
                "message": "Kolom CandlePattern tidak terbentuk."
            }), 400

        last_pattern = df_feat.iloc[-1]["CandlePattern"]

        model = load_best_model(last_pattern)

        X_last = df_feat.tail(1).drop(columns=["CandlePattern"], errors="ignore")
        prediction = model.predict(X_last)[0]

        interpretation = (
            "Harga diperkirakan naik besok"
            if prediction == 1 else
            "Harga diperkirakan turun besok"
        )

        return jsonify({
            "status": "success",
            "ticker": ticker,
            "CandlePattern": str(last_pattern),
            "prediction": int(prediction),
            "interpretation": interpretation
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)