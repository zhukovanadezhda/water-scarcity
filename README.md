# 🚱 Water Shortage Prediction at Hi!ckathon 2024

![hi-paris](assets/hiparis.jpg)

## 🔍 Overview
This repository contains the work developed by our team for the [Hi!ckathon](https://www.hi-paris.fr/hickathon/), a competition focused on AI and sustainability organized by Hi! PARIS - the Center on Data Analytics and Artificial Intelligence for Science, Business and Society created by Institut Polytechnique de Paris and HEC Paris and joined by Centre Inria de Saclay. The goal of our project was to build an AI model capable of predicting groundwater levels for French piezometric stations, with a special emphasis on the summer months. Our model uses a variety of data sources, including piezometric data, weather patterns, hydrology, water withdrawal, and economic data, to make accurate predictions. 

In addition to model development, we were tasked with considering the real-world application of our solution and projecting how it could be used in the market to address water shortages 🌍💧

## 🚀 Objective
The primary objective of the project is to:
- Build a predictive model for forecasting groundwater levels at French piezometric stations.
- Focus specifically on the summer months, as they are crucial for water resource management.
- Leverage multiple data sources, including weather, hydrology, water withdrawal, and economic data, to improve prediction accuracy.
- Explore and design a real-world application of the model to address water shortage issues.

## 👥 Our Team

![Team Picture](assets/team.png)

## 🎯 Our Approach

The target variable is categorical, with 5 balanced classes representing groundwater levels: `very low`, `low`, `average`, `high`, and `very high`. Since the data is balanced, no specific techniques for handling imbalanced data were necessary, and the models were trained to perform classification. 

The data preprocessing steps included removing columns with over 80% missing values, followed by imputing the remaining missing values with either the median or mode. Feature engineering was then performed, as detailed below. All numeric features were scaled, and the target variable was encoded as integers from 0 to 4. 

Subsequently, five models were trained and evaluated using 3-fold cross-validation, with results presented in the results section. The best-performing model was a `random forest`, which underwent grid search for hyperparameter tuning. The final F1 score of this model on the test set was 58.36%, placing the team 15th out of 60 teams.

### Feature Engineering

| **Feature**                          | **Description**                                                                                                                                             |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `day`                                | Extracted day from the `meteo_date`. Represents the day of the month.                                                                                      |
| `month`                              | Extracted month from the `meteo_date`. Represents the month of the year.                                                                                   |
| `quarter`                            | Extracted quarter from the `meteo_date`. Represents which quarter of the year (1 to 4).                                                                    |
| `year`                               | Extracted year from the `meteo_date`. Represents the year of the data point.                                                                                |
| `day_sin`                            | Sin transformation of the `day` feature. Converts the day of the month into a periodic value for modeling cyclical behavior.                               |
| `day_cos`                            | Cos transformation of the `day` feature. This, alongside `day_sin`, captures the periodicity of the day of the month.                                        |
| `month_sin`                          | Sin transformation of the `month` feature. Converts the month into a periodic value to model cyclical patterns (seasons, etc.).                            |
| `month_cos`                          | Cos transformation of the `month` feature. Works together with `month_sin` to capture the cyclical nature of months.                                        |
| `quarter_sin`                        | Sin transformation of the `quarter` feature. Captures the cyclic behavior of the four seasons in a year.                                                    |
| `quarter_cos`                        | Cos transformation of the `quarter` feature. Works together with `quarter_sin` to capture the periodic nature of quarters.                                 |
| `meteo_temperature_avg_lag_1`        | Lag feature representing the average temperature from the previous year. This helps capture long-term temperature trends.                                    |
| `meteo_rain_height_lag_1`            | Lag feature representing the rainfall from the previous year. Similar to temperature lag, this captures long-term precipitation trends.                      |
| `meteo_temperature_avg_rolling_mean_7` | Rolling mean of the average temperature over a 7-day window. This smooths out short-term fluctuations and helps capture medium-term temperature trends.      |
| `meteo_rain_height_rolling_sum_7`    | Rolling sum of the rainfall over a 7-day window. Helps to capture cumulative rainfall over a short period.                                                   |
| `temperature_wind_interaction`       | Interaction feature between average temperature and wind speed. Helps to capture the joint effect of temperature and wind on environmental conditions.        |
| `humidity_rain_interaction`          | Interaction feature between humidity and rainfall. Helps to understand how the two variables interact and affect the environment together.                  |
| `temperature_range`                  | Difference between the maximum and minimum temperature. Captures the temperature variability within a day or over time.                                      |
| `evapotranspiration_to_rain_ratio`   | Ratio of evapotranspiration to rainfall. Helps understand how the amount of water evaporated compares to the rainfall, influencing soil moisture.             |
| `altitude_difference`                | Difference between the piezo station altitude and the meteorological station altitude. Helps to capture geographic effects on environmental conditions.       |
| `cumulative_rainfall_30_days`        | Rolling sum of rainfall over a 30-day window. Captures long-term trends in precipitation.                                                                  |


## 📊 Results

| Model        | Accuracy       | F1 Score       | Precision      | Recall         | AUC-ROC        |
|--------------|----------------|----------------|----------------|----------------|----------------|
| **Random Forest** | 0.7149 ± 0.0004 | 0.7212 ± 0.0005 | 0.7231 ± 0.0010 | 0.7199 ± 0.0001 | 0.9222 ± 0.0003 |
| **XGBoost**  | 0.6261 ± 0.0014 | 0.6349 ± 0.0014 | 0.6352 ± 0.0013 | 0.6349 ± 0.0016 | 0.8821 ± 0.0007 |
| **LightGBM** | 0.5851 ± 0.0014 | 0.5928 ± 0.0015 | 0.5925 ± 0.0015 | 0.5940 ± 0.0015 | 0.8592 ± 0.0010 |
| **CNN** | 0.5223 ± 0.0048 | 0.5227 ± 0.0049 | 0.5241 ± 0.0047 | 0.5223 ± 0.0048 | 0.8342 ± 0.0027|
| **AdaBoost** | 0.3390 ± 0.0020 | 0.3411 ± 0.0025 | 0.3432 ± 0.0032 | 0.3408 ± 0.0018 | 0.6756 ± 0.0007 |


## 🖥️ Run the code

To set up the environment and install the required dependencies, use the following commands:

```bash
conda env create -f environment.yml
conda activate water-scarcity
```

Then, clone the repository and navigate to the project folder:

```bash
git clone git@github.com:zhukovanadezhda/water-scarcity.git
cd water-scarcity
```

### Preprocessing

Download the data to the `data` folder (contact us to get the data). Then run this command to get the train and test datasets:

```bash
python scripts/preprocess_data.py --path <data_file_path> [--is_train]
```
```bash
    --path        Path to the CSV data file (training or test).
    --is_train    Flag to indicate training data (optional).
```

### Models training and evaluation

After the preprocessing is completed, use one of two scripts `train_cnn.py` or `train_models.py` to train and evaluate corresponding models. 

```bash
python scripts/train_cnn.py --X_path data/X_train.csv --y_path data/y_train.csv 
```
```bash
    --X_path      Path to the CSV file containing the training features.
    --y_path      Path to the CSV file containing the training labels.
```

## 🤝 Acknowledgments

- Hi! PARIS for organizing the Hi!ckathon and providing the opportunity to work on impactful sustainability challenges 🎉
- The participants, mentors, and organizers for their valuable feedback and support during the competition.
