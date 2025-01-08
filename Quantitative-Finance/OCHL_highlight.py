import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol.
    
    :param ticker: Stock ticker (e.g., 'AAPL', 'GOOG')
    :param start_date: Start date (e.g., '2020-01-01')
    :param end_date: End date (e.g., '2023-01-01')
    :return: Pandas DataFrame containing stock data
    """
    try:
        # Fetch stock data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # If data is empty, shift the dates
        attempts = 0
        while stock_data.empty and attempts < 5:  # Try up to 5 times
            print(f"Warning: No data for {ticker} from {start_date} to {end_date}. Shifting dates...")
            start_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")  # Shift start date +1 day
            end_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")  # Shift end date -1 day
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            attempts += 1

        if stock_data.empty:
            raise ValueError(f"Unable to retrieve data for {ticker} after shifting dates.")
        
        # Clean the data: Remove rows with NaN in critical columns (Open, High, Low, Close)
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure all the relevant columns are of float type
        stock_data[['Open', 'High', 'Low', 'Close']] = stock_data[['Open', 'High', 'Low', 'Close']].astype(float)
        
        # Flatten MultiIndex columns if necessary (i.e., remove the "Ticker" part from the columns)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)  # Get the first level (Open, High, Low, Close)
        
        # Ensure the DataFrame index is of datetime type (mplfinance requires this)
        stock_data.index = pd.to_datetime(stock_data.index)
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def highlight_large_candles(stock_data, threshold=0.01):
    """
    Identify candles with more than a specified percentage change between Open and Close.
    
    :param stock_data: DataFrame containing stock data (must have Open, High, Low, Close)
    :param threshold: Percentage change threshold to highlight (default: 1%).
    :return: List of color labels corresponding to the candles in stock_data
    """
    # Initialize a list with default 'none' color for all candles
    highlight_colors = ['none'] * len(stock_data)

    # Skip the first candle and iterate through the stock data
    for i in range(1, len(stock_data)):
        open_price = stock_data['Open'].iloc[i]
        close_price = stock_data['Close'].iloc[i]

        # Calculate percentage change between Open and Close
        pct_change = (close_price - open_price) / open_price

        # Check if the change exceeds the threshold
        if abs(pct_change) > threshold:
            highlight_colors[i] = 'green' if pct_change > 0 else 'red'  # Green for positive, red for negative change

    return highlight_colors

def plot_ohlc_chart(stock_data, highlight_colors):
    """
    Plot OHLC (Open, High, Low, Close) chart using Matplotlib (mplfinance) and highlight large candles.
    
    :param stock_data: DataFrame containing stock data (must have Open, High, Low, Close)
    :param highlight_colors: List of colors corresponding to the candles to highlight
    """
    # Ensure 'highlight_colors' length matches the length of stock data
    if len(highlight_colors) != len(stock_data):
        raise ValueError("Length of highlight_colors does not match the number of data points.")
    
    # Convert the 'highlight_colors' to a format mplfinance can use
    highlight = mpf.make_addplot(stock_data['Close'], type='scatter', markersize=10, marker='o', color=highlight_colors)
    
    # Plot OHLC chart using mplfinance
    mpf.plot(stock_data, type='candle', style='charles', title="OHLC Chart", ylabel="Price (USD)", volume=True,
             addplot=highlight)

def main():
    # Ask for user input
    ticker = input("Enter the stock ticker symbol (e.g., 'AAPL'): ")

    # Ask for start date (no validation)
    start_date = input("Enter the start date (YYYY-MM-DD): ")

    # Ask for end date (no validation)
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Get stock data with shifted dates if necessary
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        # Highlight large candles
        highlight_colors = highlight_large_candles(stock_data)

        # Plot OHLC chart with highlighted candles
        plot_ohlc_chart(stock_data, highlight_colors)
    else:
        print("Failed to retrieve stock data.")

# Run the program
if __name__ == "__main__":
    main()