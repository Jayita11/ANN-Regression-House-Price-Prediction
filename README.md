# House Price Prediction using ANN

This project aims to predict house prices using an Artificial Neural Network (ANN) model. The project includes data preprocessing, model training, and deployment using a Streamlit app.

![image](https://github.com/user-attachments/assets/279cc55d-cdf0-4420-a04e-ee39c4601915)


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Building](#model-building)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
  - [Data Cleaning](#data-cleaning)
  - [Feature Engineering](#feature-engineering)
  - [Model Optimization](#model-optimization)
  - [Prediction Interpretation](#prediction-interpretation)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Prediction](#model-prediction)
  - [Deployment](#deployment)
- [License](#license)

## Overview

This project develops an artificial neural network model to perform regression on a housing dataset to predict the median house value. It includes loading and preprocessing the dataset, handling categorical variables, and building and training a multi-layer perceptron model using TensorFlow Keras. The model is compiled and trained on the preprocessed training set, and its predictive performance is evaluated on the held-out test set.

## Dataset

The dataset used in this project is the California Housing Prices dataset. The features include various attributes of houses such as location, age, rooms, bedrooms, population, households, median income, and proximity to the ocean.

## Methodology

### Data Collection

- The dataset is collected from a CSV file named `housing.csv`.

### Data Preprocessing

- Missing values are handled by dropping rows with missing data.
- Categorical variables are encoded using numerical mappings.
- Features are scaled using Min-Max scaling.

### Splitting Data

Divide the dataset into training and testing sets. The training set is used to train the ANN, while the testing set is used to evaluate its performance.

### Model Building

- An artificial neural network with two hidden layers is built using TensorFlow Keras.

### Model Training

- The model is trained using the RMSprop optimizer and mean squared error loss function.
- Early stopping and tensor board callbacks are used to improve training.

The following graphs show the training and validation loss and mean absolute error (MAE) over the epochs:

#### Training and Validation Loss
The first graph below shows the training and validation loss over 100 epochs. The loss decreases significantly during the initial epochs and then stabilizes, indicating that the model is learning and converging.

  ![3](https://github.com/user-attachments/assets/dbc27b75-eee0-48aa-926b-098bbc5dac6d)

#### Training and Validation MAE
The second graph displays the mean absolute error (MAE) for both training and validation sets over the epochs. The MAE decreases as the training progresses, showing that the model's predictions are becoming more accurate.
  
  ![4](https://github.com/user-attachments/assets/63dee67a-c366-4cc9-b898-7da1ba7acbd2)

#### Actual vs Predicted Prices
The final graphs compare the actual house prices with the predicted prices for both the training and validation datasets. The points should ideally align along the red diagonal line, indicating accurate predictions.

  ![5](https://github.com/user-attachments/assets/39506735-8cdc-4f4c-9d1f-e6e0d771e6c0)


### Model Evaluation

- The model's performance is evaluated using mean absolute error and mean squared error metrics.

![9](https://github.com/user-attachments/assets/97ba7064-1375-4f63-9d69-7f54f54fd93a)



## Results

- The trained model can predict house prices with a reasonable degree of accuracy.
- The final model and scaler are saved as `model_ann_reg.h5` and `min_max_scaler.pkl`.

## Deployment
Deploy the model using a Streamlit app (app.py). The app allows users to input house data and get price predictions. To run the app, execute the following command:

https://ann-regression-house-price-prediction-a9fzp8yemstma6tah8sdtq.streamlit.app/

![Animation_2](https://github.com/user-attachments/assets/d61ec3a8-34d5-41b9-8e78-2778c878a98d)

  
This starts a web server and opens the app in the default web browser, enabling interaction with the model for house price predictions.
## Challenges and Solutions

### Data Cleaning

- Handled missing values by dropping rows with missing data.

### Feature Engineering

- Categorical feature `ocean_proximity` is mapped to numerical values.

### Model Optimization

- Various hyperparameters such as the number of units in hidden layers, dropout rates, and learning rates are tuned for optimal performance.
- Early stopping is used to prevent overfitting.

### Prediction Interpretation

- The model is used to predict house prices based on user inputs, providing insights into the influence of various features on house prices.

## Installation

To run this project, you need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/house-price-prediction.git](https://github.com/Jayita11/ANN-Regression-House-Price-Prediction
    cd house-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

