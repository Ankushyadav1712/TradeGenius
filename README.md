# 📊 Phase 1: Data Collection & Exploration
## AI Market Trend Analysis Project

This folder contains all the deliverables for **Phase 1** of the AI Market Trend Analysis project, focusing on data collection and exploratory data analysis.

---

## 📋 Phase 1 Objectives

1. **Data Collection**: Collect historical stock market data from Yahoo Finance
2. **Data Exploration**: Perform exploratory data analysis (EDA) on collected data
3. **Data Quality**: Ensure data quality and completeness
4. **Documentation**: Document findings and insights

---

## 📁 Folder Structure

```
phase1_submission/
│
├── 📂 src/
│   ├── __init__.py
│   └── data_collector.py      # Main data collection script
│
├── 📂 notebooks/
│   └── 01_data_exploration.ipynb  # Jupyter notebook for EDA
│
├── 📂 data/
│   └── raw/                    # Raw stock data (generated after running collector)
│       └── stock_data.csv
│
├── 📋 requirements.txt         # Python dependencies
└── 📚 README.md               # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Stock Data

Run the data collection script:

```bash
python src/data_collector.py
```

This will:
- Download 5 years of historical data for major tech stocks (AAPL, GOOGL, MSFT, AMZN, TSLA)
- Save the data to `data/raw/stock_data.csv`
- Display a summary of collected data

### 3. Explore the Data

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or if using JupyterLab:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

---

## 📊 Data Collection Details

### Stock Symbols Collected
- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc. (Google)
- **MSFT** - Microsoft Corporation
- **AMZN** - Amazon.com Inc.
- **TSLA** - Tesla Inc.

### Data Fields
Each record contains:
- **Date**: Trading date
- **Symbol**: Stock ticker symbol
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume
- **Daily_Return**: Percentage change from previous day
- **Price_Range**: High - Low
- **Volume_MA_20**: 20-day moving average of volume

### Data Period
- **Default**: 5 years of daily data
- **Frequency**: Daily (1d)
- **Source**: Yahoo Finance API (via yfinance library)

---

## 🔧 Usage Examples

### Basic Usage

```python
from src.data_collector import StockDataCollector

# Initialize with default stocks
collector = StockDataCollector()

# Fetch 5 years of data
data = collector.fetch_stock_data(period="5y", interval="1d")

# Get summary
collector.get_data_summary()

# Save to file
collector.save_data("data/raw/stock_data.csv")
```

### Custom Stocks

```python
# Collect data for specific stocks
custom_stocks = ['NVDA', 'META', 'NFLX']
collector = StockDataCollector(symbols=custom_stocks)
data = collector.fetch_stock_data(period="2y")
```

### Get Recent Data

```python
# Get last 30 days of data
recent_data = collector.get_latest_data(days=30)
```

---

## 📈 Data Exploration Notebook

The `01_data_exploration.ipynb` notebook includes:

1. **Data Loading**: Load and inspect the collected data
2. **Basic Statistics**: Summary statistics and data quality checks
3. **Price Visualization**: Interactive charts showing price movements
4. **Volume Analysis**: Trading volume distribution and patterns
5. **Returns Analysis**: Daily returns distribution and statistics
6. **Volatility Analysis**: Rolling volatility calculations and visualization
7. **Key Insights**: Summary of findings and observations

---

## 📊 Expected Output

After running the data collection script, you should see:

```
🚀 AI Market Trend Analysis - Data Collection Demo
=======================================================
📊 StockDataCollector initialized with 5 symbols:
   AAPL, GOOGL, MSFT, AMZN, TSLA

🔄 Fetching 5y of 1d data...
   📈 Getting data for AAPL...
   ✅ Got 1258 days of data for AAPL
   ...
   
✅ Successfully fetched data:
   📅 Date range: 2019-01-02 to 2024-01-15
   📊 Total rows: 6,290
   🏢 Symbols: 5

📊 DATA SUMMARY
============================================================
📅 Date Range: 2019-01-02 to 2024-01-15
📈 Symbols: AAPL, AMZN, GOOGL, MSFT, TSLA
📊 Total Records: 6,290
🗓️  Trading Days: 1,258
...
```

---

## 🔍 Data Quality Checks

The data collector automatically performs:
- ✅ Missing value detection
- ✅ Data type validation
- ✅ Date range verification
- ✅ Symbol validation
- ✅ Basic statistical summary

---

## 📝 Notes

- **Internet Connection Required**: Data collection requires an active internet connection
- **API Rate Limits**: Yahoo Finance API may have rate limits; if errors occur, wait a few minutes and retry
- **Data Freshness**: Data is collected in real-time from Yahoo Finance
- **File Size**: The generated CSV file is typically 500KB - 2MB depending on the period

---

## 🐛 Troubleshooting

### Issue: "No data found for [SYMBOL]"
- **Solution**: Check if the stock symbol is correct and the market is open
- **Alternative**: Try a different time period or symbol

### Issue: Import errors
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Issue: FileNotFoundError when running notebook
- **Solution**: Make sure you've run `python src/data_collector.py` first to generate the data file

---

## 📚 Next Steps (Phase 2)

After completing Phase 1, the next phase will focus on:
- Feature Engineering: Creating technical indicators
- Data Preprocessing: Scaling and normalization
- Model Preparation: Preparing data for machine learning

---





