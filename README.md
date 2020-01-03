# Financial Modeling

### Introduction
When tasked with predicting stock prices, we decided it would be important to consider current methods being used by professionals in the field of algorithmic trading as well as research being done in this area. After our preliminary research was complete, we converged on specific methods to featurize our data and specific models we would try to implement in order to predict the stock prices of the top 200 technology companies on Nasdaq for a variable amount of days into the future. We mainly consider extending ML techniques: first moving from simple linear regression to SVR’s, and then exploring Ridge Regression, Lasso Regression, Elastic Net Regression,and Random Forests, all of which are supported in literature.

We believe that when predicting next-day prices, using broad market and industry indicator features (in addition to single-stock features) will improve upon the performance achieved using single-stock features. Intuitively, this makes sense; in the short run, such as next-day performance, general market outlook may weigh heavily on investors’ appetite for new investments. We do not expect the variance of these results to differ significantly from single stock prediction models due to the stock market being heavily dependent on industry performance as well as individual company performance. In terms of noise, industry indicators such as competitors’ performance or other sector-level indicators could lead to more noise in the results because of the additional, possibly insignificant data.

### Dataset
For this project, we chose to use data provided by Quandl for stock price data such as the adjusted prices of specific stocks. We first gathered a list of stock tickers from Nasdaq’s official dataset of stocks split by industry. We decided to focus on stocks in the Technology industry due to their reliable trends and present-day relevance. We picked the top 200 from this list and used the tickers to get the stock price data from Quandl. We cleaned the data in a similar fashion to the starter file provided for this project in order to only keep the Adjusted open, close, high and low prices of a one day period in each stock’s data frame. We also calculate the percent change in price from the open and close prices per day and store in a separate column of data. We then store each clean data frame in a dictionary mapped by stock ticker.

### Featurization - Single Stock
Initially, our dataset only had data on the prices on each day over the common time frame we worked with. In order to make meaningful decisions from this data, we decided to featurize the data with specific, industry-recognized stock and industry level indicators of future stock price, most of which are outlined in our initial project proposal. We looked at three main types of indicators: volatility, momentum, and trend. We chose these based on their prevalent usage in firms and research around the world. After selecting features important to us, we used the Python library ta (technical analysis) to assist us in calculating these features.

For volatility and trend indicators, we chose to use, respectively, the average true range (ATR) and average directional movement index (ADX) because of their common occurrence in industry-level single stock predictions. ATR was also designed to support daily prices and measures volatility over 14 day periods, which makes it even more applicable to our data. ATR additionally only measures an absolute value of distance between the previous day close price and the current high and low prices, not considering direction, which is why we also need ADX to provide insight on the direction of the stock’s price movement. ADX uses the ATR to calculate a scaled and smoothed moving average value that indicates the strength of a stock’s trend. In this sense, it is a lagging indicator, which requires the data to have an established trend before ADX can provide a reliable signal. This is why we also use ATR as a feature in the scenario that we cannot rely on ADX. In addition to signalling the strength of a trend, ADX is renowned for assisting in market timing methods such as a buy signal when the ADX peaks and starts to decline. With this strategy you would sell when the ADX stops falling and goes flat.

Other volatility indicators include Bollinger bands, of which we used the high band indicator to signal when a stock is performing above its expected range of volatility. The Bollinger band is also useful to us in the calculation of the relative strength index (RSI), a momentum indicator. RSI is useful in indicating whether a stock, index, or other investment is overbought or oversold. It is calculated as the inverse of the average gain over average loss during a defined period of time, scaled by a constant to result in a score from 0 to 100. The RSI is most helpful in a non-trending environment in which the the value of a stock or other financial instrument fluctuates between a range of two prices. RSI is particularly useful when it reaches the extremes of its 0 to 100 range. The last single stock indicator we looked at is Williams %R, which is the inverse of another stochastic oscillator. Essentially this indicator gives the relationship between the close price and the high-low range for a given period in the past (default period is usually 14 days for daily stock price predictions). This indicator also, on the extremes of its range from 0 to -100, is useful to tell whether a stock is oversold or overbought.

### Featurization - Industry Level
Since the purpose of our project was to try and improve single stock predictions by adding industry level features, we explored a number of industry and market-level features. We added these features to the set of single-stock, for each stock (matching by date and normalizing). We quickly realized that there is little access to index-level features for free on Quandl. We were able to find the following industry level features (from Quandl provider “NASDAQOMX”), which track general tech-sector stock performance.
* NASDAQ-100 Ex-Tech Sector (NDXX)
* NASDAQ-100 Technology Sector (NDXT) 
* NASDAQ-100 Technology Sector Total Return (NTTR)

At the market level, we were able to find the following (from Quandl provider “NASDAQOMX”). These indices track general NASDAQ performance, in America, across all stocks, and across smaller sets of companies (by market cap):
* NASDAQ N America Index (NQNA)
* NASDAQ US All Market Index (NQUSA)
* NASDAQ US 1500 Index (NQUSS1500)
* NASDAQ US 450 Index (NQUSM450)
* NASDAQ US 300 Index (NQUSL300)
* NASDAQ US Small Cap Index (NQUSS)
* NASDAQ US Large Cap Index (NQUSL)
* NASDAQ US Mid Cap Index (NQUSM)



