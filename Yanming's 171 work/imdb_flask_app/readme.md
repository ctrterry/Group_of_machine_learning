# IMDb Rating Predictor 

imdb-flask_app/
├── app.py # Main Flask app
├── rf_model.pkl # Trained Random Forest model (required)
├── templates/
│ └── index.html # Web form UI
├── requirements.txt # Python dependencies
└── README.md # Project overview

This is a Flask web app that predicts IMDb movie ratings using a Random Forest model trained on movie metadata.

## Features
- User input for runtime, budget, Facebook likes, etc.
- Genre selection
- Outputs predicted IMDb rating
- Trained model file: `rf_model.pkl`

## How to Run

```bash
pip install flask
pip install numpy
pip install pandas
pip install scikit-learn

python app.py

then, run http://127.0.0.1:5000 at browser
