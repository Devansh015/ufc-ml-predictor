# 🥊 UFC ML Fight Predictor

A machine learning project that predicts the winner of a UFC fight based on historical fight data, designed to beat my gambling addicted friends 
## 🔍 Overview

This project uses historical UFCStats data to train machine learning models that estimate win probabilities for upcoming matchups. Users provide two fighters (Red vs Blue corner), and the model outputs a predicted winner along with confidence scores.

All features are derived strictly from **past fights** to prevent data leakage.

## 🧠 Core Features

- End-to-end ML pipeline (data → features → model → prediction)
- User inputs **two fighters** to generate a matchup prediction
- Outputs:
  - Predicted winner
  - Win probability per fighter
- Engineered matchup (delta) features between fighters

## 🖥️ Frontend WIP

The project will include a lightweight frontend that:
- Allows users to **select fighters from searchable dropdowns**
- Sends the selected matchup to the prediction engine
- Displays results in a clear, user-friendly format
- To be used for for future fight predictions

The frontend will interface directly with the trained model via a simple API or local prediction script.

## ⚙️ Tech Stack

- Python  
- Pandas / NumPy  
- scikit-learn  
- LightGBM  

## 📌 Status

🚧 Actively in development — backend ML pipeline complete, frontend integration in progress.

## ⚠️ Disclaimer

This project is for educational and research purposes only.  
Predictions are not betting or financial advice.
