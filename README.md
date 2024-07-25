# House Price Prediction using ANN

This project aims to predict house prices using an Artificial Neural Network (ANN) model. The project includes data preprocessing, model training, and deployment using a Streamlit app.

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

  ![3](https://github.com/user-attachments/assets/dbc27b75-eee0-48aa-926b-098bbc5dac6d)
  
  ![4](https://github.com/user-attachments/assets/63dee67a-c366-4cc9-b898-7da1ba7acbd2)

  ![5](https://github.com/user-attachments/assets/39506735-8cdc-4f4c-9d1f-e6e0d771e6c0)


### Model Evaluation

- The model's performance is evaluated using mean absolute error and mean squared error metrics.

![9](https://github.com/user-attachments/assets/97ba7064-1375-4f63-9d69-7f54f54fd93a)



## Results

- The trained model can predict house prices with a reasonable degree of accuracy.
- The final model and scaler are saved as `model_ann_reg.h5` and `min_max_scaler.pkl`.

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

## Usage

### Model Prediction

To predict house prices using the pre-trained model, run the `predict_house_price.py` script:

```sh
python predict_house_price.py
