import pandas as pd
import matplotlib.pyplot as plt

# Load data here. Change via the file path.
file_path = "Data/prices_round_1_day_0.csv" 
df = pd.read_csv(file_path, sep=';')

# Basic info
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Unique products
products = df["product"].dropna().unique()
print("\nProducts found:", products)

# Plot mid price over time for each product
plt.figure(figsize=(12, 6))
for product in products:
    sub = df[df["product"] == product]
    plt.plot(sub["timestamp"], sub["mid_price"], label=product)

plt.title("Mid Price Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot bid/ask spread over time 
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

plt.figure(figsize=(12, 6))
for product in products:
    sub = df[df["product"] == product]
    plt.plot(sub["timestamp"], sub["spread"], label=product)

plt.title("Bid-Ask Spread Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram of mid prices
plt.figure(figsize=(10, 6))
for product in products:
    sub = df[df["product"] == product]
    plt.hist(sub["mid_price"].dropna(), bins=30, alpha=0.5, label=product)

plt.title("Distribution of Mid Prices")
plt.xlabel("Mid Price")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Profit and loss over time 
plt.figure(figsize=(12, 6))
for product in products:
    sub = df[df["product"] == product]
    plt.plot(sub["timestamp"], sub["profit_and_loss"], label=product)

plt.title("Profit and Loss Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Profit and Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()