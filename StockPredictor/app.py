import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "TSLA", "JPM", "JNJ", "V", "PG", "UNH"
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3-vl:235b-cloud"

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("stock-oracle")


# ---------- DATA ----------
def get_stock_data(ticker, period="6mo"):
    logger.info("Fetching price data for %s", ticker)
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if hist.empty:
            logger.warning("No price data found for %s", ticker)
            return None
        return hist["Close"].dropna().values
    except Exception as e:
        logger.exception("Error while fetching data for %s: %s", ticker, e)
        return None


# ---------- ML ----------
def train_model(prices):
    if len(prices) < 20:
        logger.warning("Not enough data to train model")
        return None, None, None

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(5, len(scaled_prices)):
        X.append(scaled_prices[i - 5:i])
        y.append(scaled_prices[i])

    model = LinearRegression()
    model.fit(np.array(X), np.array(y))

    logger.info("ML model trained successfully")
    return model, scaler, np.array(X[-1])


def predict_next_day(model, scaler, last_5):
    if model is None or scaler is None or last_5 is None:
        return None
    pred_scaled = model.predict([last_5])[0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)


# ---------- AI NEWS ANALYSIS ----------
def parse_ai_response(text):
    """
    Tries to parse JSON from Ollama.
    Falls back to simple text extraction if needed.
    """
    cleaned = text.strip()

    # Remove markdown code fences if the model adds them
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    data = {}
    try:
        data = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except Exception:
                data = {}

    summary = data.get("summary", cleaned[:300] if cleaned else "No summary available.")
    sentiment = str(data.get("sentiment", "neutral")).lower()
    recommendation = str(data.get("recommendation", "Hold")).title()
    reason = data.get("reason", "")

    if sentiment not in ["positive", "neutral", "negative"]:
        sentiment = "neutral"

    if recommendation not in ["Buy", "Hold", "Avoid"]:
        recommendation = "Hold"

    return {
        "summary": summary,
        "sentiment": sentiment,
        "recommendation": recommendation,
        "reason": reason
    }


def get_ai_analysis(ticker):
    logger.info("Requesting AI news analysis for %s", ticker)

    prompt = f"""
You are a stock news analyst.

Analyze {ticker} using recent news and market sentiment.

Return ONLY valid JSON with these keys:
{{
  "summary": "short summary under 80 words",
  "sentiment": "positive" or "neutral" or "negative",
  "recommendation": "Buy" or "Hold" or "Avoid",
  "reason": "one short sentence explaining the recommendation"
}}

Be careful: make the recommendation based mostly on news sentiment and company outlook.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=25
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "")
        ai = parse_ai_response(text)

        logger.info(
            "AI result for %s -> sentiment=%s, recommendation=%s",
            ticker, ai["sentiment"], ai["recommendation"]
        )
        return ai

    except Exception as e:
        logger.exception("AI analysis failed for %s: %s", ticker, e)
        return {
            "summary": "AI analysis unavailable right now.",
            "sentiment": "neutral",
            "recommendation": "Hold",
            "reason": "AI service failed."
        }


# ---------- ANALYSIS ----------
def analyze_stock(ticker):
    ticker = ticker.upper().strip()
    logger.info("Starting full analysis for %s", ticker)

    prices = get_stock_data(ticker)
    if prices is None or len(prices) < 20:
        logger.warning("Analysis skipped for %s due to insufficient data", ticker)
        return None

    model, scaler, last_5 = train_model(prices)
    if model is None:
        return None

    current_price = float(prices[-1])
    predicted_price = predict_next_day(model, scaler, last_5)
    if predicted_price is None:
        logger.warning("Prediction failed for %s", ticker)
        return None

    ai = get_ai_analysis(ticker)

    predicted_change_pct = ((predicted_price - current_price) / current_price) * 100.0
    sentiment_score = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
    }.get(ai["sentiment"], 1)

    recommendation_score = {
        "Buy": 3,
        "Hold": 2,
        "Avoid": 0
    }.get(ai["recommendation"], 2)

    # AI-driven first, price trend is secondary
    final_score = (recommendation_score * 100) + (sentiment_score * 10) + predicted_change_pct

    invest_decision = "Yes" if ai["recommendation"] == "Buy" else "No"

    logger.info(
        "Finished analysis for %s | current=%.2f predicted=%.2f decision=%s",
        ticker, current_price, predicted_price, ai["recommendation"]
    )

    return {
        "ticker": ticker,
        "current_price": current_price,
        "predicted_price": float(predicted_price),
        "predicted_change_pct": float(predicted_change_pct),
        "summary": ai["summary"],
        "sentiment": ai["sentiment"],
        "recommendation": ai["recommendation"],
        "reason": ai["reason"],
        "invest_decision": invest_decision,
        "score": float(final_score)
    }


def find_best_stock():
    logger.info("Scanning all stocks to find the best one")

    results = []
    max_workers = min(4, len(STOCKS))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(analyze_stock, ticker): ticker for ticker in STOCKS}

        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info("Collected result for %s", ticker)
            except Exception as e:
                logger.exception("Stock scan failed for %s: %s", ticker, e)

    if not results:
        logger.error("No stock results available")
        return None

    best = max(results, key=lambda x: x["score"])
    logger.info("Best stock found: %s", best["ticker"])
    return best


# ---------- ROUTES ----------
@app.route("/")
def home():
    logger.info("Serving landing page")
    return render_template("index.html")


@app.route("/api/best_stock")
def best_stock():
    best = find_best_stock()
    if not best:
        return jsonify({"error": "Could not evaluate stocks"}), 500

    return jsonify({
        "ticker": best["ticker"],
        "current_price": best["current_price"],
        "predicted_price": best["predicted_price"],
        "predicted_change_pct": best["predicted_change_pct"],
        "summary": best["summary"],
        "sentiment": best["sentiment"],
        "recommendation": best["recommendation"],
        "reason": best["reason"],
        "invest_decision": best["invest_decision"]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "").upper().strip()

    if not ticker:
        logger.warning("Predict request missing ticker")
        return jsonify({"error": "No ticker provided"}), 400

    logger.info("Received predict request for %s", ticker)
    result = analyze_stock(ticker)

    if not result:
        return jsonify({"error": "Invalid ticker or not enough data"}), 400

    return jsonify({
        "ticker": result["ticker"],
        "current_price": result["current_price"],
        "predicted_price": result["predicted_price"],
        "predicted_change_pct": result["predicted_change_pct"],
        "summary": result["summary"],
        "sentiment": result["sentiment"],
        "recommendation": result["recommendation"],
        "reason": result["reason"],
        "invest_decision": result["invest_decision"]
    })


if __name__ == "__main__":
    logger.info("Starting Stock Oracle on port 8080")
    app.run(debug=True, host="0.0.0.0", port=8080)