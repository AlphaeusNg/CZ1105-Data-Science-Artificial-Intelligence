# File explanation
Datasets file contains the datasets used for the exploratory analysis and linear regression model.
There are one .py file and one .ipynb file, they are the same. Just in different format.

# CZ1105-Data-Science-Artificial-Intelligence
A DSAI project from the NTU module CZ1105 titled "Are we able to predict stock prices (from different industries) using Covid-19 variables?”.

Used Alpha Vantage API to call stock market prices where we store it into excel files.
Used Covid Tracking API to call covid dataset where we also store it into excel files.

Used Scikit-learn library to build a univariate and multivariate linear regression model to test our prediction accuracy.
Used NumPy, Seaborn, Matplotlib to clean, organise, display our data.

## Background
We were curious to find out if there was any relation between these 2 different variables. So our initial hypothesis was this - As covid-19 worsens, the prices in the stock market
would drop. We came to this hypothesis because it seemed natural for market sentiments to be reflective of how Covid was doing.

Out of the large pool of data available to us, we chose 6 stocks from different
sectors of the stock market, namely:

- Apple (AAPL) and Tesla (TSLA) representing the technology industry
- Halliburton (HAL) from the energy industry
- PayPal (PYPL) from finance
- Singapore Airlines (SINGF) from the aviation industry
- United Health (UNH) from the healthcare industry
  
And for the covid-19 API, we also chose 6 variable, mainly:
- Total number of Test
- Total Increase in the number of Test
- Total number of Positive Results
- Total Increase in the number of Positive Results
- Total number of deaths
- Total increase in the number of deaths

As we found that these variables were the most reflective of the covid situation. Before we started on our analysis, we needed to align the timeline for our 2 datasets. We found that there were more constraints with regards to the Covid dataset such as the dates before April 2020 had mostly empty data slots, and the last entry being march 2021. Therefore we chose to align both datasets with the time frame from 1st April 2020 to 31st December 2020 and isolated the corresponding data from the dataset respectively before starting any analysis.

## Project workflow

### Data visualisation and hypothesis testing
Plotted covid-19 variable and stock price with respect to the time. This is done so that we can observe any relation or trends as time passes. 
Generally, we observed an increase in stock prices as the covid variables increase in number as well, which actually contradicts our first hypothesis - That as covid worsens, the prices in the stock market would drop. 
It is interesting to note that the only sector which had an inversely proportional relationship to covid was the transport sector. Our intuitive guess was due to the halt on international travelling, SINGF, which is the Singapore airline stock, reflected that.

After visual inference, we supported our observation with statistics and plotted a correlation heatmap and corresponding pair plot for each stock against all the covid-19 variables.
We foudn out that there was some strong and weak correlation amongst the variables. But we found that the variables with the strongest correlation were, ‘Total number test case’ and ‘total number of deaths’, which indicates that they were the most important variables in predicting stock prices.

With exploratory data analysis completed, we tackled our main problem of predicting the stock prices using covid-19 variables. To do this, we checked that the variables were all numeric. Once that was done, we decided to utilize the regression technique that we have previously learnt, to predict the stock prices. We chose to use regression because regression analysis uses a defined dependent variable that we hypothesized as being influenced by one or several independent variables. And since we have inferred previously that there is some correlation between the datasets, we believe that regression technique was appropriate in reaching our goal.

### Training model
The data  was split randomly into 75% train data and 25% test data. We created our uni-variate linear regression model and subsequently used it to predict the stock prices by using the test variables on our model. We did this for all the stocks against all the covid-19 variables as shown. We collated the explained variance and mean square error for comparison and concluded which models were the best for each stock based on their higher R^2 and lower MSE. Chiefly:
- AAPL_Prices vs Total_Testing
- TSLA vs Total_Testing
- HAL vs Total_Deaths
- PYPL vs Total_Deaths
- SINGF vs Death_Increased
- UNH vs Total_Deaths

To further explore and learn new techniques to further improve our models in predicting stock prices. We used the multi-variate linear regression model, and it was highly suitable as it factors in multiple variables.

We created a multi-variate linear regression model using the same process as the uni-variate model. We randomly split the data into a 3:1 ratio for train and test data, however, this time, we factored in all the covid-19 variables. We wanted to see if the multi-variate model was more accurate and had a better goodness of fit based on the explained variance and mean square error compared to the uni-variate linear regression model we previously plotted. And indeed, we saw an improvement in terms of higher explained variance and lower MSE. MV yielded much more accurate data across all the different stocks.

## Conclusion
We were able to predict stock prices with marginal error. We also found out that the covid variables `Total number of test cases` and `total number of deaths` were indeed the most important variables in predicting stock prices as previously observed in our correlation heatmap and pairplot. 

We also learned that the multi-variate linear regression model definitely had a better ability in predicting stock prices. And in the end we are able to conclude that our first hypothesis about an inverse relationship of stock prices and covid-19 was actually false, but instead, it has some sort of a directly proportional relationship instead. 

## Further improvements
More data points could have been used so that the sample size can be increased for better accuracy and in turn, better credibility.
