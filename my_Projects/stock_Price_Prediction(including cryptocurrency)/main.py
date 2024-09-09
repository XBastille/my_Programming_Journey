import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import ta

class StockDataVisualizer:
    def __init__(self, company_name):
        self.company_name = company_name
        self.stock_data = self.download_stock_data()

    def download_stock_data(self):
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=25*365)
        stock_data = yf.download(self.company_name, start=start_date, end=end_date)
        print(yf.Ticker(self.company_name).info)
        stock_data.drop(columns=["Open", "High", "Low", "Adj Close", "Volume"], inplace=True)
        return stock_data

    def calculate_price_change(self):
        close_price=self.stock_data["Close"].iloc[-1]
        previous_close=self.stock_data["Close"].iloc[-2]
        price_change=close_price-previous_close
        percentage_change=(price_change/previous_close)*100
        return price_change, percentage_change

    def plot_historical_data(self, start_date=None, end_date=None):
        df = self.plot_technical_indicators_2(start_date, end_date)
        price_change, percentage_change=self.calculate_price_change()
        plt.figure(figsize=(14,7))
        plt.plot(df["Close"], label="Historical Close Prices")
        title = f"Historical Close Prices for {self.company_name} "
        if price_change > 0:
            title+=f"+{price_change:.2f} (+{percentage_change:.2f}%)"
        else:
            title+=f"{price_change:.2f} ({percentage_change:.2f}%)"
        plt.title(title)
        self.plot_technical_indicators_28_10("Price")

    def plot_technical_indicators(self, start_date=None, end_date=None):
        df = self.plot_technical_indicators_2(start_date, end_date)
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"]=macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_histogram"] = macd.macd_diff()
        bollinger=ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["Bollinger_hband"]=bollinger.bollinger_hband()
        df["Bollinger_lband"]=bollinger.bollinger_lband()
        
        plt.figure(figsize=(14,7))
        plt.plot(df["Close"], label="Close Price")
        plt.plot(df["SMA50"], label="50-Day SMA")
        plt.plot(df["SMA200"], label="200-Day SMA")
        plt.fill_between(df.index, df["Bollinger_hband"], df["Bollinger_lband"], color="grey", alpha=0.3, label="Bollinger Bands")
        self.plot_technical_indicators_28(' Technical Indicators', "Price")
        
        plt.figure(figsize=(14,4))
        plt.plot(df["RSI"], label="RSI", color="purple")
        plt.axhline(70, color="red", linestyle="--")
        plt.axhline(30, color="green", linestyle="--")
        self.plot_technical_indicators_28(' RSI', "RSI")
        
        plt.figure(figsize=(14,7))
        plt.plot(df["MACD"], label="MACD", color="blue")
        plt.plot(df["MACD_signal"], label="MACD Signal", color="red")
        plt.bar(df.index, df["MACD_histogram"], label="MACD Histogram", color="green", alpha=0.3)
        self.plot_technical_indicators_28(' MACD', "MACD")
        
    def plot_technical_indicators_2(self, start_date, end_date):
        result = self.stock_data.copy()
        if start_date:
            result = result[result.index >= pd.to_datetime(start_date)]
        if end_date:
            result = result[result.index <= pd.to_datetime(end_date)]
        return result

    def plot_technical_indicators_28(self, arg0, arg1):
        plt.title(f"{self.company_name}{arg0}")
        self.plot_technical_indicators_28_10(arg1)

    def plot_technical_indicators_28_10(self, arg0):
        plt.xlabel("Date")
        plt.ylabel(arg0)
        plt.legend()
        plt.grid()
        plt.show()

    def run(self):
        start_date = input("Enter the start date for the plots (YYYY-MM-DD) or press Enter to use the full date range: ")
        end_date = input("Enter the end date for the plots (YYYY-MM-DD) or press Enter to use the full date range: ")
        start_date = start_date or None
        end_date = end_date or None
        self.plot_historical_data(start_date, end_date)
        self.plot_technical_indicators(start_date, end_date)

if __name__ == "__main__":
    company_name = input("Enter the company ticker (e.g., TSLA): ")
    visualizer = StockDataVisualizer(company_name)
    visualizer.run()
