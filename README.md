# Predictive Modelling with Stocks Data
This project contains a Flask application that trains different ML Models on stock prices and then show it on a plot.

## Description
This flask based tool compares the predictive modelling power of different Machine/Deep Learning models. For dataset, i have opening, high, low and closing values for different stocks like AAL, AAP, AAPL. Then to train different models, tool provides with a bunch of options like Random Forests, K Nearest Neighbors, SVM, LSTM etc.. Models are implemented using scikit-learn and keras. Performance of each chosen model is shown using Chart.js library.

## Requirements
This project requires python3. To install required libraries, run the following command.
```
pip3 install -r requirements.txt
```

## Running the tool
To run the tool, use the following command
```
python3 app.py
```

## Demo

![Alt Text](https://github.com/asad1996172/Predictive-Modelling-on-Stock-Prices/blob/master/demo.gif)
