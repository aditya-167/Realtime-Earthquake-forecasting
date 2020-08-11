# Realtime Earthquake Predictor application 

A realtime earthquake predictor web app with google maps API, that forecasts earthquake possible epicenters and places in window of next 7 days.

![web-app](https://github.com/aditya-167/Realtime-Earthquake-forecasting/blob/master/Images/application.jpg)

Web app link : [http://srichaditya3098.pythonanywhere.com/](http://srichaditya3098.pythonanywhere.com/)
## Contents

   * Project Overview
   * Problem Statement and approach to solution
   * Metrics 
   * Dataset 
   * Exploratory Data Analysis and Data processing
   * Model implementation
   * Improvement and evaluation
   * Prediction and web application
   * Improvement and conclusion
   * Deployment web app
   * Instructions to run the project
   * acknowledgement

### Project Overview
Countless dollars and entire scientific careers have been dedicated to predicting where and when the next big earthquake will strike. But unlike weather forecasting, which has significantly improved with the use of better satellites and more powerful mathematical models, earthquake prediction has been marred by repeated failure due to highly uncertain conditions of earth and its surroundings.
Now, with the help of artificial intelligence, a growing number of scientists say changes in the way they can analyze massive amounts of seismic data can help them better understand earthquakes, anticipate how they will behave, and provide quicker and more accurate early warnings. This helps in hazzard assessments for many builders and real estate business for infrastructure planning from business perspective. Also many lives can be saved through early warning. This project aims a simple solution to above problem by predicting or forecasting likely places to have earthquake in next 7 days. For user-friendly part, this project has a web application that extracts live data updated every minute by USGS.gov and predicts next likely place world wide to get hit by an earthquake, hence a realtime solution is provided.

### Problem Statement and approach to solution
Anticipating seismic tremors is a pivotal issue in Earth science because of their overwhelming and huge scope outcomes. The goal of this project is to predict where likely in the world and on what dates the earthquake will happen. Application and impact of the project​ includes potential to improve earthquake hazard assessments that could spare lives and billions of dollars in infrastructure and planning. Given geological locations, magnitude and other factors in dataset from https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php for 30 days past which is updated every minute, we predict or forecast 7 days time in future that is yet to come, the places where quake would likely happen. The web app uses Google maps api to predict places where earthquake might occur.

### Metrics

The problem addressed above is about binary classification, `Earthquake occur = 1` and `Earthquake not occur = 0` and with these prediction we try to locate co-cordinates corrosponding to the predictions and display it on the google maps api web app. More suitable metrics for binary clsssification problems are **ROC (Reciever operator characteristics), AUC (Area Under Curve), Confusion matrix for Precision, recall, accuracy and sensitivity**. One important thing about choosing metrics and model is what exactly we need from predictions and what not. To be precise, we need to **minimize or get less False negative predictions** since we dont want our model to predict as `0` or `no earthquake occured` at particular location when in reality it had actually happend as this is more dangerous than the prediction case in which prediction is `true/1` or `earthquake occured` but in reality it did not because its always **better safe than sorry!!!**. Hence apart from `roc_auc score`, I have considered
`Recall` as well for evaluation and model selection with `higher auc_roc score and recall`, where `recall = (TP/TP+FN)`.

### Dataset

Real time data that updates every minute on https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php for past 30 days. Below is the feature description of the dataset with 22 features and 14150 samples at the time of training.

* time ---------------------- Time when the event occurred. Times are reported in milliseconds since the epoch 
* latitude ------------------- Decimal degrees latitude. Negative values for southern latitudes.
* longitude ------------------ Decimal degrees longitude. Negative values for western longitudes.
* depth ---------------------- Depth of the event in kilometers.
* mag ------------------------ Magnitude of event occured.
* magType -------------------- The method or algorithm used to calculate the preferred magnitude
* nst ------------------------ The total number of seismic stations used to determine earthquake location.
* gap ------------------------ The largest azimuthal gap between azimuthally adjacent stations (in degrees).
* dmin ----------------------- Horizontal distance from the epicenter to the nearest station (in degrees).
* rms ------------------------ The root-mean-square (RMS) travel time residual, in sec, using all weights.
* net ------------------------- The ID of a data source contributor for event occured.
* id -------------------------- A unique identifier for the event. 
* types ----------------------- A comma-separated list of product types associated to this event.
* place ----------------------- named geographic region near to the event.
* type ------------------------ Type of seismic event.
* locationSource -------------- The network that originally authored the reported location of this event.
* magSource ------------------- Network that originally authored the reported magnitude for this event.
* horizontalError ------------- Uncertainty of reported location of the event in kilometers.
* depthError ------------------ The depth error, three principal errors on a vertical line.
* magError -------------------- Uncertainty of reported magnitude of the event.
* magNst ---------------------- The total number of seismic stations to calculate the magnitude of earthquake.
* status ---------------------- Indicates whether the event has been reviewed by a human.



### Exploratory Data Analysis and Data preprocessing

This part is best explained in project walkthrough notebooks `Data/ETL_USGS_EarthQuake.ipybn` or `Data/ETL_USGS_EarthQuake.html`.
Finally the cleaned data for prediction is stored in database file `Data/Earthquakedata.db` using sql engine.

**Note** : only for project walkthrough purpose cleaned data is stored in database but for realtime analysis, in `Webapp/main.py` flask app, we extract data on the go without storing. This make sures we get realtime data any day when web app is requested by any user.

### Model implementation

As for the model selection, I have tried with Boosting algorithms for classification problem.

1. Adaboost classifier with estimator as DecisionTreeClassifier

2. Adaboosr classifier with estimator as RandomForestClassifier

3. Finally I tried Xgboost algorithm.

model selection was based on Evaluation on `roc_auc score` and `recall` and hyperparameter tunning.
A better walkthrough is mentioned with great detail in `models/Earthquake-predictor-ML-workflow.ipybn` or `models/Earthquake-predictor-ML-workflow.html`.

### Improvement and evaluation 

* I have used gridsearch CV for improving model and hyperparameter tunning on Adaboost classifier with base estimators as `DecisionTreeClassifier` and `RandomForestClassifier`.
* Using the same hyper parameters I trained XGBoost. As mentioned above, metrics for evaluation is `roc_auc score` and `recall`.

***DecisionTreeClassifier adaboost**

![DecisionTreeClassifier evaluation](https://github.com/aditya-167/Realtime-Earthquake-forecasting/blob/master/Images/DecisionTree.jpg)

1. With **adaboost decision tree classifier** and hyper parameter tunning, we get area under curve (score) = 0.8867
2. higher the auc score, better is the model since it is better at distinguishing postive and negative classes.
3. Make a note here that we get from **confusion matrix**, `False negative = 42`and `Recall score =0.7789`. We need this value apart from auc score that we will analyze later when we have tested with diffferent models below


***RandomForesClassifier adaboost**

![RandomForestClassifier evaluation](https://github.com/aditya-167/Realtime-Earthquake-forecasting/blob/master/Images/RandomForest.png)

1. Below is the auc score for **adaboost RandomForest classifier** with 0.916 which is slightly lower than Decision tree classifier
2. Moreover when we look at **confusion matrix**, `False Negative=38` and `Recall score = 0.8' can be observed which is slightly higher than recall score of decision tree. Thus performs better than decision tree adabooost 
