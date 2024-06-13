# Machine Learning with XGBoost and Streamlit 

Machine Learning (ML) and Artificial Intelligence (AI) have become hot topics in recent years, partly due to large language models like ChatGPT. With a growing number of tools available for creating machine learning models, it can be challenging to know where to start. The choice depends on various factors, including the type of data involved. While everyone jokes about AI world domination, smart technology professionals are busy mastering machine learning, unlocking endless possibilities, and leading the charge in innovation. Among the tools garnering attention in this sphere are [Streamlit](https://streamlit.io/) and [XGBoost](https://xgboost.ai/) (eXtreme Gradient Boosting), which we will explore further.

## Example: Stock Market Prediction

As an example, we will be using stock market data provided by the Yahoo Finance library in python. 

```python
import yfinance as yf
def load_data(stock_symbol):
    data = yf.download(stock_symbol)
    return data
```

## Machine Learning Models

At its core, machine learning involves training a model on a sample dataset to identify relationships between the target variable (the variable you aim to predict) and other variables using algorithms such as regression or neural networks. Here, we use `XGBRegressor`, which implements a gradient boosting algorithm, and define the closing price as the target variable.

```python
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xg_reg.fit(scaled_train_data, train_data['Close'])
```

These relationships, often challenging for humans to discern, are effectively identified within a specific calculable error margin. Once a model is trained, it can be used to make predictions on new, unseen data. For example, we can predict the closing stock price given the open price, high price, low price and volume for a given stock. 

```python
input_data = np.array([[open_price, high_price, low_price, volume]])
input_data_scaled = scaler.transform(input_data)
prediction = xg_reg.predict(input_data_scaled)[0]
```

For developers, this process typically involves creating an API from the model or developing an application that accepts user inputs to generate predictions using the model.

```python
# Displaying input fields with labels and default values
stock_symbol = st.text_input("Stock Symbol", value="GOOGL", max_chars=5)
prediction_date = st.date_input("Prediction Date", value=datetime.today())
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=110.0)
low_price = st.number_input("Low Price", value=90.0)
volume = st.number_input("Volume", value=1000000)
```

Thus, by retrieving new data about a stock provided by the user in streamlit, we can predict the close price using XGBoost.

## Why XGBoost and Streamlit?

Among the myriad of machine learning libraries, XGBoost stands out for its exceptional speed and performance. When coupled with Streamlit, a powerful tool for creating interactive web applications in Python, developers can build robust and user-friendly solutions effortlessly. While XGBoost is compatible with multiple programming languages, including Python, R, Java, and Scala, leveraging Python's XGBoost library alongside the Streamlit web framework allows us to rapidly create simple machine learning applications.

[Explore some Streamlit apps created by other developers.](https://streamlit.io/gallery)

## Model Evaluation: How Accurate Is Our Prediction?

A critical step in machine learning is evaluating our predictions to assess how accurately it identifies the target variable, in this case, the closing price of a stock. Here, we employ RMSE (Root Mean Squared Error) on our test set, which constitutes a subset of the initial dataset.

```python
test_predictions = xg_reg.predict(scaled_test_data)
rmse = np.sqrt(mean_squared_error(test_data['Close'], test_predictions))
```

For instance, if a user inputs information for an Apple stock and receives a prediction of $150.49 for the closing price, the accuracy of this prediction hinges on the RMSE value. A RMSE of $0.006 would indicate high accuracy, whereas $12.38 would signify lower accuracy.

## Conclusion

XGBoost is widely used across industries, exemplified by tech leaders like Netflix and Uber. It excels in predictive analytics, forecasting outcomes such as stock prices, sales figures, and service demand. Streamlit enables swift prototyping, allowing users to harness XGBoost-generated models effectively. While achieving millions from stock markets through machine learning may be ambitious, numerous impressive ML projects seamlessly integrate with websites and mobile apps. Leveraging these tools in our applications enhances their capabilities significantly!


