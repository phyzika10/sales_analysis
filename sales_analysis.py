# Step 1: Import Required Libraries
import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import sqlite3       # For SQL database operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.ensemble import RandomForestRegressor  # For predictive modeling
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation

# Step 2: Load the Dataset
# Replace 'superstore_sales.csv' with the path to your dataset
# Specify the correct encoding (e.g., 'latin1', 'ISO-8859-1', or 'cp1252')
df = pd.read_csv('superstore_sales.csv', encoding='latin1')

# Step 3: Data Cleaning
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Step 4: Store Data in a SQL Database
# Connect to SQLite database (creates it if it doesn't exist)
conn = sqlite3.connect('sales_data.db')
cursor = conn.cursor()

# Create a table to store sales data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        order_id TEXT PRIMARY KEY,
        order_date DATE,
        sales REAL,
        profit REAL,
        category TEXT,
        sub_category TEXT,
        region TEXT
    )
''')
conn.commit()

# Insert data into the table
df.to_sql('sales', conn, if_exists='replace', index=False)

# Step 5: Perform SQL Queries for Analysis
# Query 1: Total sales by category
query1 = '''
    SELECT category, SUM(sales) AS total_sales
    FROM sales
    GROUP BY category
    ORDER BY total_sales DESC
'''
category_sales = pd.read_sql(query1, conn)
print("Total Sales by Category:\n", category_sales)

# Query 2: Monthly sales trends
query2 = '''
    SELECT strftime('%Y-%m', "Order Date") AS month, SUM(sales) AS total_sales
    FROM sales
    GROUP BY month
    ORDER BY month
'''
monthly_sales = pd.read_sql(query2, conn)
print("Monthly Sales Trends:\n", monthly_sales)

# Query 3: Most profitable sub-categories
query3 = '''
    SELECT "Sub-Category", SUM(profit) AS total_profit
    FROM sales
    GROUP BY "Sub-Category"
    ORDER BY total_profit DESC
'''
profitable_subcategories = pd.read_sql(query3, conn)
print("Most Profitable Sub-Categories:\n", profitable_subcategories)

# Step 6: Exploratory Data Analysis (EDA)
# Plot 1: Monthly sales trends
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['month'], monthly_sales['total_sales'], marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Plot 2: Total sales by category
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='total_sales', data=category_sales)
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

# Plot 3: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Profit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Sales and Profit')
plt.show()

# Step 7: Feature Engineering
# Extract features from 'Order Date'
df['year'] = df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.month
df['day_of_week'] = df['Order Date'].dt.dayofweek

# Encode categorical variables (one-hot encoding)
df = pd.get_dummies(df, columns=['Category', 'Region'], drop_first=True)

# Drop unnecessary columns
df = df.drop(['Order Date', 'Order ID'], axis=1)

# Step 8: Build a Predictive Model
# Split the data into features (X) and target (y)
X = df.drop('Sales', axis=1)
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'Model Performance:\nRMSE: {rmse}\nR-squared: {r2}')

# Step 9: Save the Model
import joblib
joblib.dump(model, 'sales_prediction_model.pkl')

cursor.execute('DROP TABLE IF EXISTS sales')

# Step 10: Close the SQL Connection
conn.close()
