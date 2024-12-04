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

## 🎯 Our solution

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

## 🤝 Acknowledgments

- Hi! PARIS for organizing the Hi!ckathon and providing the opportunity to work on impactful sustainability challenges 🎉
- The participants, mentors, and organizers for their valuable feedback and support during the competition.
