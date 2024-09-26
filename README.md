# Iris Flower Classification with Machine Learning

This repository contains the code and resources for building a **Machine Learning model** to classify the species of iris flowers based on their measurements. The dataset includes three species: **Setosa**, **Versicolor**, and **Virginica**, and the model is trained to distinguish between them using features such as sepal length, sepal width, petal length, and petal width.

Additionally, a **Streamlit app** has been created to interactively classify the iris species, and this app has been deployed on Hugging Face Spaces. You can explore the app [here](https://huggingface.co/spaces/zafermbilen/iris-flower-classification).

## Live Demo

You can try out the Iris Flower Classification model live on [Hugging Face Spaces](https://huggingface.co/spaces/zafermbilen/iris-flower-classification).

## Overview

The **Iris Flower Classification** problem is a well-known dataset often used in machine learning classification tasks. The goal is to classify the iris flowers into one of the three species:

- Setosa
- Versicolor
- Virginica

Using four features (sepal length, sepal width, petal length, petal width), we can train a classifier that can predict the species of a new iris flower.

This project demonstrates the entire machine learning pipeline:

- Data Preprocessing
- Model Training
- Model Evaluation
- Deployment of the model using **Streamlit**.

## Dataset

The dataset used for this project is the **Iris Dataset**, which consists of 150 samples with 4 features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each sample is labeled with one of the three species of iris flowers. You can explore the dataset more on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) or use the dataset provided by the `Scikit-learn` library.

## Model

The machine learning model has been implemented using Python and the following libraries:

- **Scikit-learn**: For loading the dataset and building the classification model
- **Streamlit**: For deploying an interactive app

The model used is a **Support Vector Machine (SVM)**, but this can be easily swapped for other algorithms like Decision Trees, Logistic Regression, or K-Nearest Neighbors.

## App

The project also includes a simple web-based interface using **Streamlit** to interact with the model. You can input your own measurements, and the app will classify the iris species for you.

You can find the deployed app on Hugging Face Spaces [here](https://huggingface.co/spaces/zafermbilen/iris-flower-classification).

### How it works:

1. The user provides input for the four flower measurements.
2. The trained model predicts the species of the iris flower.
3. The prediction is displayed on the web page.
