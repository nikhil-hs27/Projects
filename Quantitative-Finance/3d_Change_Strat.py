import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
        
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


def add_percentage_stage(stock_data):
    """
    Adds a new column 'Percentage Change' to the stock data DataFrame.
    """
    stock_data['Percentage Change'] = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']) * 100
    return stock_data

def add_direction_labels(stock_data):
    """
    Adds a new column 'Direction' to the stock data DataFrame based on the percentage change.
    """
    def label_direction(pct_change):
        if pct_change > 1:
            return 'pos'  # Positive change (greater than 1%)
        elif pct_change < -1:
            return 'neg'  # Negative change (less than -1%)
        else:
            return 'no_trade'  # No significant change

    stock_data['Direction'] = stock_data['Percentage Change'].apply(label_direction)
    return stock_data

def simulate_trading(stock_data, initial_capital=1000):
    """
    Simulates trading based on the provided strategy:
    - Three consecutive positive days trigger a short trade.
    - Three consecutive negative days trigger a long trade.
    """
    capital = initial_capital
    position = None  # Can be 'long' or 'short'
    entry_price = 0
    exit_range_high = 0
    exit_range_low = 0
    trade_day = 0  # Track the day we are on in the trade

    for i in range(2, len(stock_data)):  # Start from index 2 (since we need at least 3 days for the condition)
        # Get direction of the last 3 days
        last_3_days = stock_data['Direction'].iloc[i-2:i+1].values
        
        # If there are three consecutive 'pos' days, take a short trade
        if last_3_days[0] == 'pos' and last_3_days[1] == 'pos' and last_3_days[2] == 'pos' and position is None:
            position = 'short'
            entry_price = stock_data['Close'].iloc[i]
            exit_range_high = entry_price * 1.02  # 1% above the entry price for short
            exit_range_low = entry_price * 0.99   # 1% below the entry price for short
            trade_day = i
            print(f"Opening short trade at {entry_price} on day {stock_data.index[i]}")
        
        # If there are three consecutive 'neg' days, take a long trade
        elif last_3_days[0] == 'neg' and last_3_days[1] == 'neg' and last_3_days[2] == 'neg' and position is None:
            position = 'long'
            entry_price = stock_data['Close'].iloc[i]
            exit_range_high = entry_price * 1.01  # 1% above the entry price for long
            exit_range_low = entry_price * 0.98   # 1% below the entry price for long
            trade_day = i
            print(f"Opening long trade at {entry_price} on day {stock_data.index[i]}")
        
        # If we're in a position, check the exit conditions
        if position is not None and i > trade_day:
            # Ensure current_price is a scalar value
            current_price = stock_data['Close'].iloc[i]  # Extract the scalar value
            
            # Exit the short trade if the price goes above the exit range (loss) or below the exit range (profit)
            if position == 'short':
                if current_price > exit_range_high:  # Exit condition for loss (price went above 1% from entry)
                    capital += (entry_price - current_price) * (capital / entry_price)  # Loss calculation for short
                    print(f"Closing short trade at {current_price} (loss) on day {stock_data.index[i]}")
                    position = None  # Reset position
                    entry_price = 0  # Reset entry price
                    trade_day = 0  # Reset trade day
                elif current_price < exit_range_low:  # Exit condition for profit (price went below 1% from entry)
                    capital += (entry_price - current_price) * (capital / entry_price)  # Profit calculation for short
                    print(f"Closing short trade at {current_price} (profit) on day {stock_data.index[i]}")
                    position = None  # Reset position
                    entry_price = 0  # Reset entry price
                    trade_day = 0  # Reset trade day

            # Exit the long trade if the price goes below the exit range (loss) or above the exit range (profit)
            elif position == 'long':
                if current_price < exit_range_low:  # Exit condition for loss (price went below 1% from entry)
                    capital += (current_price - entry_price) * (capital / entry_price)  # Loss calculation for long
                    print(f"Closing long trade at {current_price} (loss) on day {stock_data.index[i]}")
                    position = None  # Reset position
                    entry_price = 0  # Reset entry price
                    trade_day = 0  # Reset trade day
                elif current_price > exit_range_high:  # Exit condition for profit (price went above 1% from entry)
                    capital += (current_price - entry_price) * (capital / entry_price)  # Profit calculation for long
                    print(f"Closing long trade at {current_price} (profit) on day {stock_data.index[i]}")
                    position = None  # Reset position
                    entry_price = 0  # Reset entry price
                    trade_day = 0  # Reset trade day
                
    return capital

def main():
    ticker = input("Enter the stock ticker symbol (e.g., 'AAPL'): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        # Add percentage change and direction labels
        stock_data = add_percentage_stage(stock_data)
        stock_data = add_direction_labels(stock_data)

        # Simulate trading
        final_capital = simulate_trading(stock_data)
        print(f"Final capital: ${final_capital:.2f}")
    else:
        print("Failed to retrieve stock data.")

if __name__ == "__main__":
    main()