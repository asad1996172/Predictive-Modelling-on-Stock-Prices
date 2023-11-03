# Predictive Modelling with Stocks Data
The repository titled "Predictive Modelling on Stock Prices" is a Flask-based application designed to train various machine learning models on stock price data and visualize their predictions. The application allows users to compare the performance of different models, such as Random Forests, K Nearest Neighbors, SVM, and LSTM, implemented using scikit-learn and Keras. The predictions are plotted using the Chart.js library, providing an interactive way to analyze the predictive capabilities of each model.

Features:
- Model Training: Supports training multiple machine learning models on stock price data.
- Visualization: Utilizes Chart.js to plot the predictions of each model for comparison.
- Data Handling: Includes a dataset with stock prices (e.g., GOOG_30_days.csv) to train the models.
- User Interaction: Offers a web interface for users to select stocks and models for prediction.

Technical Details:
- Web Framework: Flask
- Machine Learning: scikit-learn, Keras
- Data Visualization: Chart.js
- Programming Language: Python


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
