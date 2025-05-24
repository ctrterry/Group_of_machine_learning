IMDb Movie Rating Prediction Project
====================================

Overview
--------
This project aims to predict IMDb movie ratings using various metadata features such as budget, runtime, genres, cast, crew, writers, and directors. We use machine learning models to evaluate how well these features can predict user-rated movie scores.


Models Used
-----------
1. Random Forest Regressor
   - Mean Squared Error (MSE): 0.492
   - R² Score: 0.483

2. Linear Regression
   - Mean Squared Error (MSE): 0.601
   - R² Score: 0.368

Model Comparison
----------------
Random Forest performed better than Linear Regression by capturing nonlinear patterns in the data. Linear Regression underfit the data and did not capture important feature interactions.

Important Features (Random Forest)
----------------------------------
- numVotes
- runtime
- budget
- revenue
- genre
- actor_count
- writer/director counts

How to Use
----------
1. Open the notebook `imdb_full_model_training.ipynb` in Jupyter or VS Code.
2. Run the cells to reproduce the feature engineering, training, and visualizations.
3. Compare the two models visually and numerically.

Conclusion
----------
Random Forest is a more effective model for predicting movie ratings in this context. Linear Regression, while interpretable, is not powerful enough to model the complexity in the dataset.

Future Work
-----------
- Test more advanced models such as XGBoost or Neural Networks
- Add text-based features (plot summaries, reviews)
- Deploy as a web app with user input

