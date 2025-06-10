# Group Study of Machine Learning



1, How to run the flask
cd "Yanming_Directory/imdb_flask_app"
python app.py

2, How to run with IMDB data as ANN, Random Forest, and Linear Regression
cd "Sasha_Directory/actors/ann"
python ann.py


3, Run with IMDB + social Data as ANN, Random Forest, and Linear Regression
cd "Terry_Directory/New_work"
python ann.py

4. Detailed breakdown of the project structure, and explanation of all the files used/created related to EXP dataset, check ./Sasha_Directory/actors/README.md



## Research Analysis Paper

[My Reseach Analysis Paper IEEE Version](./terry_report/main.pdf)

---

## Update Algorithem Report

### Navroop Report
DNN Improvement Model Performance:
- R²: 0.4554
- MSE: 0.9535
- MAE: 0.7035

### Terry:
1, Linear-Regression (Basedline)
- MSE: 0.840
- R²: 0.229

2, MLPRegressor (ANN) Evaluation Report
Test set size: 1061
- MSE: 0.9111
- R²: 0.2030

3, K-Folder + Random-Forest
- Average MSE: 0.6832 ± 0.0476
- Average R²: 0.4256 ± 0.0253

4, Random Forest Evaluation Report
Test set size: 1061
- MSE: 0.6186
- RMSE: 0.7865
- MAE: 0.5747
- R²: 0.4589

### Yanming:

1. Random Forest Regressor
   - Mean Squared Error (MSE): 0.492
   - R² Score: 0.483

2. Linear Regression
   - Mean Squared Error (MSE): 0.601
   - R² Score: 0.368
---

## Week 7  
Notes: This is Week 7, and just a reminder — our group project is due in **2 weeks (June 2)**.

### TODO List:

#### Sasa
Train 3 models based on experience (actor score, director score, genre…)
Report performance metrics with a bar graph for all models
Ablation studies and Hyperparameter tuning

#### Terry
Train Linear Regression model and add the data to the metrics for the other 2 models
Provide a paragraph on EDA and data description for Facebook likes usages related to directors and actors.

- Done
- Conclusion, the IMDb_feature is linearly relationship with rating(Predict). Due to higher MSE and Lower R^2
- Also, the Linear_regression is not good model to predict IMDB feature vs rating.
- code was save into. terry_report/local_code/Liearly_regression.ipynb

#### Yanming
(ASAP) Update the dataset with one-hot encoding for genre and language
Learn Flask
Create a structure for the website created with Flask which fits the structure of our project

#### Navroop
(ASAP) Push cleaned and imputed writers data/scores to Github
Find unique literatures and write 3-4 sentences which answer each of the following questions, provide citations and references which follow IEEE formatting :
How did we decide on initial features (already in the MidQuarterReport)
Why did we decide to use Random Forest, Linear Regression and ANN? (check previous studies)
Why did we use MSE, R2, F1 score and MCC to evaluate performance?
How many actors impact movie performance (ideally find evidence for 3 main actors and 7 main actors)?
What is an ablation study and why is it important?

#### Safwan
(ASAP) Push cleaned and imputed directors data/scores to Github
Write a short paragraph explaining the choice of hyperparameter for ANN, and the choice of values for each
Batch size (32, 64)
Hidden layer node count (32, 64, 128)
Learning rate (0.1, 0.01)
Activation functions (Tanh, ReLU)
Momentum (0.6, 0.9)
Regularization Parameter
Write a few sentences explaining why K-fold cross-validation is used (5 folds)

---

## Week 6

Notes: This is Week 6, and just a reminder — our group project is due in **3 weeks (June 2)**.

### Terry's Final Progress Report

In this report, I focused on comparing two machine learning models:

- **Artificial Neural Network (ANN)**
- **Random Forest Tree**

### Data Preparation and Model Comparison

- Some informations I have write a report "main_not_done.pdf"
- And summary file into the foler of "Jupyter"
- and some plot into the folder of "Important_plot"
- and i'm using dataset inot the folder of "Data"
- finally, my code running result intot the folder of "Process_Report"

#### Referecnce as what I'm used
- https://www.kaggle.com/c/tmdb-box-office-prediction/data
- https://developer.imdb.com/non-commercial-datasets/
- https://www.the-numbers.com/custom-search
- https://github.com/sundeepblue/movie_rating_prediction
- https://github.com/jingkunzler211/IMDB_prediction
- https://medium.com/%40jingkunzler211/choosing-the-best-regression-model-imdb-movie-rating-prediction-3298fb11b6d
- etc. 
---

## Week 5 

Notes: This is Week 5, and just a reminder — our group project is due in **4 weeks (June 2)**.

#### TODO List:
- Since everyone has different experience with machine learning, and we want to maximize our enjoyment with this project, I have an idea.

#### About week 6 group plan:
- We can split the work as individually. Each person can find an interesting paper or favorit method (i.e ANN, CNN, etc.) and train your own model. Then, next week on Thursday (Or ~~ ?), we can compare with our models and share our final results.

- About datasets:
- Since I'm already uploaded the merge_datasets on GitHub, but you are still welcome to re-clean the data with your own.

- Finally, good luck with all midterms.

---

## Group Submition Paper

- 1-page-proposal-group5.pdf 
- mid_quarter_report.pdf


