# 🧠📈 Optiver - Trading at the Close: GRU-based Price Movement Forecasting

Welcome to my solution for the [Optiver: Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) Kaggle competition.

In this high-stakes prediction challenge, the goal is to forecast **auction price movements** of stocks using ultra-high-frequency order book and auction data — a real-world, time-critical trading scenario.

This project uses a **GRU-based deep learning model** integrated with **Polars for lightning-fast data processing**, and fine-tuned using **Optuna**. The final solution is compatible with Kaggle’s **live prediction API**.

---
 <p align="center">
  <img src="https://i.imgur.com/SuYWQ6e.png" width="500">

## 🎯 Objective

Forecast a stock’s **final auction price movement** using historical features extracted from:
- Order book snapshots
- Auction and transaction history
- Time-series derived metrics

📏 Evaluation Metric: **Mean Absolute Error (MAE)**  
🎯 Target: `target = wap - reference_price`

---
### 🎯 Target Distribution

 <p align="center">
  <img src="https://i.imgur.com/B1aVk13.png" width="500">
  
---

## 🧱 Pipeline Overview

| Stage                      | Description |
|---------------------------|-------------|
| 📦 **Data Loading**        | Loaded 100M+ rows with `Polars` for speed and memory efficiency |
| 🧹 **Cleaning & Imputation** | Dropped `near_price`, `far_price`, handled nulls & infinities |
| 📊 **Feature Engineering** | Custom price/spread/imbalance metrics, slopes, lags, and volatility |
| 🧩 **Sequence Modeling**   | GRU (Gated Recurrent Unit) with PackedSequence & variable time steps |
| 🧪 **Hyperparameter Tuning** | Used Optuna for optimizing GRU depth, dropout, learning rate, etc. |
| 🚀 **Live Submission**     | Compatible with `optiver2023.iter_test()` for real-time evaluation |

---

## 🧠 Model Architecture

- Input: Normalized sequences of engineered features
- Model: Multi-layer GRU + dense head
- Loss: `SmoothL1Loss` for robustness
- Optimizer: AdamW

<!-- Optional image -->
<!-- <p align="center">
  <img src="images/gru_model.png" width="400">
</p> -->

---

## 🧪 Evaluation Snapshot

| Model            | MAE (CV) | MAE (LB) | Notes                            |
|------------------|----------|----------|----------------------------------|
| GRU + Polars     | 0.00132  | 0.00138  | Tuned with full engineered set   |

---


## 🛠️ Tech Stack

- `Polars` – ultra-fast dataframe ops for time series
- `PyTorch` – custom GRU with Packed Sequences
- `Optuna` – Bayesian hyperparameter optimization
- `Matplotlib`, `Seaborn` – EDA & performance visuals
- `Kaggle API` – live evaluation loop

---

## 🧠 Key Learnings

- Data pipelines matter: **Polars > pandas** for scale.
- **GRU is ideal for sparse, autocorrelated time-series** problems.
- **Auto-tuning beats intuition** for hyperparameters.
- Designing for **live inference compatibility** from day 1 saves time later.

---

## 📬 Contact

Made with 💹 by [Justin Varghese](https://github.com/blacckbeard4)  
Let’s connect if you're into **ML for finance**, **streaming analytics**, or **Kaggle problem-solving**.

---
