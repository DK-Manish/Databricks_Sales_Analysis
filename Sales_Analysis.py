# Databricks notebook source
# Load the Products Data 

# File location and type
products_file_path = "/FileStore/tables/Products.csv"
file_type = "csv"

# Load CSV into a DataFrame
df_products = spark.read.format(file_type) \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(products_file_path)

# Show the DataFrame
display(df_products)

# COMMAND ----------

# Load the Sales Data 

# File location and type
sales_file_path = "/FileStore/tables/Sales.csv"

# Load CSV into a DataFrame
df_sales = spark.read.format(file_type) \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(sales_file_path)

# Show the DataFrame
display(df_sales)

# COMMAND ----------

# Performing a selective join (avoiding duplicates)

# Perform join on ProductID, selecting only necessary columns from df_products
df_joined = df_sales.join(
    df_products.select("ProductID", "Product Name", "Category", "Brand", "Supplier"),
    on="ProductID",
    how="left"
)

# Show the cleaned and merged DataFrame
display(df_joined)


# COMMAND ----------

# check for Null Values

from pyspark.sql.functions import col, when, count

# Count null values in each column
df_null_counts = df_joined.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df_joined.columns
])

# Show null counts
df_null_counts.show()

# COMMAND ----------

# Adding a Derived Column - Total Sales Amount
# Total Price = Quantity x Price

from pyspark.sql.functions import expr

# Add Total Sales Amount column
df_final = df_joined.withColumn("Total Sales Amount", expr("Quantity * Price"))

# Show the updated DataFrame
display(df_final)


# COMMAND ----------

# Calculate Total Order Value per Customer

from pyspark.sql.functions import sum

# Aggregate total order value per customer
df_customer_orders = df_final.groupBy("CustomerID").agg(
    sum("Total Sales Amount").alias("Total Order Value")
)

# Show the result
display(df_customer_orders)

# COMMAND ----------

# Calculate Total Sales per Brand

# Aggregate total sales amount by Brand
df_brand_sales = df_final.groupBy("Brand").agg(
    sum("Total Sales Amount").alias("Total Sales by Brand")
)

# Show the result
display(df_brand_sales)

# COMMAND ----------

# Calculate Total Sales per Product

# Aggregate total sales amount by Product Name
df_product_sales = df_final.groupBy("Product Name").agg(
    sum("Total Sales Amount").alias("Total Sales by Product")
)

# Show the result
display(df_product_sales)

# COMMAND ----------

# Calculate Total Sales Per Month

# Extract Month from Order Date

from pyspark.sql.functions import month

# Extract month from Order Date
df_final = df_final.withColumn("Order Month", month("Order Date"))

# Show updated DataFrame
display(df_final)

# COMMAND ----------

# Calculate Total Sales by Month

# Aggregate total sales amount by Order Month
df_monthly_sales = df_final.groupBy("Order Month").agg(
    sum("Total Sales Amount").alias("Total Sales by Month")
).orderBy("Order Month")

# Show the result
display(df_monthly_sales)

# COMMAND ----------

# Best-Selling Product (Top Product by Sales)

from pyspark.sql.functions import desc

# Find the best-selling product by revenue
df_top_product = df_product_sales.orderBy(desc("Total Sales by Product"))

# Show the result
display(df_top_product)

# COMMAND ----------

# Most Frequent Customers (Highest Order Count)

# Count total orders per customer
df_customer_orders_count = df_final.groupBy("CustomerID").count().orderBy(desc("count"))

# Show the result
display(df_customer_orders_count)

# COMMAND ----------

# Most Popular Product (Highest Quantity Sold)

# Find the product with the highest total quantity sold
df_popular_product = df_final.groupBy("Product Name").agg(
    sum("Quantity").alias("Total Quantity Sold")
).orderBy(desc("Total Quantity Sold"))

# Show the result
display(df_popular_product)

# COMMAND ----------

#  Region with the Highest Sales

# Find total sales by customer region
df_region_sales = df_final.groupBy("Customer Region").agg(
    sum("Total Sales Amount").alias("Total Sales by Region")
).orderBy(desc("Total Sales by Region"))

# Show the result
display(df_region_sales)

# COMMAND ----------

# Monthly Growth Rate in Sales

from pyspark.sql.window import Window
from pyspark.sql.functions import lag

# Define a window partitioned by Order Month
window_spec = Window.orderBy("Order Month")

# Add previous month sales column
df_monthly_sales = df_monthly_sales.withColumn(
    "Previous Month Sales", lag("Total Sales by Month").over(window_spec)
)

# Calculate Month-over-Month Growth Rate
df_monthly_sales = df_monthly_sales.withColumn(
    "MoM Growth (%)",
    ((df_monthly_sales["Total Sales by Month"] - df_monthly_sales["Previous Month Sales"])
     / df_monthly_sales["Previous Month Sales"]) * 100
)

# Show the result
display(df_monthly_sales)

# COMMAND ----------

# Function to Get Top N Products by Sales

from pyspark.sql.functions import desc

def get_top_products(df, top_n=5):
    return df.groupBy("Product Name").agg(
        sum("Total Sales Amount").alias("Total Sales by Product")
    ).orderBy(desc("Total Sales by Product")).limit(top_n)

# Get top 5 products
df_top_products = get_top_products(df_final, 5)

# Show result
display(df_top_products)

# COMMAND ----------

# Function to Get Monthly Sales with Growth Rate

from pyspark.sql.window import Window
from pyspark.sql.functions import lag

def get_monthly_sales(df):
    df_monthly_sales = df.groupBy("Order Month").agg(
        sum("Total Sales Amount").alias("Total Sales by Month")
    ).orderBy("Order Month")

    # Define window for lag function
    window_spec = Window.orderBy("Order Month")

    # Add previous month sales column
    df_monthly_sales = df_monthly_sales.withColumn(
        "Previous Month Sales", lag("Total Sales by Month").over(window_spec)
    )

    # Calculate Month-over-Month Growth Rate
    df_monthly_sales = df_monthly_sales.withColumn(
        "MoM Growth (%)",
        ((df_monthly_sales["Total Sales by Month"] - df_monthly_sales["Previous Month Sales"])
         / df_monthly_sales["Previous Month Sales"]) * 100
    )

    return df_monthly_sales

# Get monthly sales report
df_monthly_sales_report = get_monthly_sales(df_final)

# Show result
display(df_monthly_sales_report)


# COMMAND ----------

# Function to Find Top N Customers by Spending

def get_top_customers(df, top_n=5):
    return df.groupBy("CustomerID").agg(
        sum("Total Sales Amount").alias("Total Order Value")
    ).orderBy(desc("Total Order Value")).limit(top_n)

# Get top 5 customers
df_top_customers = get_top_customers(df_final, 5)

# Show result
display(df_top_customers)

# COMMAND ----------

# Store Data in Delta Lake for Better Performance

# Remove invalid characters from column names
for col_name in df_final.columns:
    new_col_name = col_name.replace(" ", "_").replace(",", "").replace(";", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("\n", "").replace("\t", "").replace("=", "")
    df_final = df_final.withColumnRenamed(col_name, new_col_name)

# Save it in Delta Lake
df_final.write.format("delta").mode("overwrite").save("/mnt/delta/sales_data")

# COMMAND ----------

# Query Delta Table Efficiently
# Instead of reading from CSV, we can directly read from Delta Lake
# This allows fast and optimized queries without reloading CSVs every time

df_delta = spark.read.format("delta").load("/mnt/delta/sales_data")
display(df_delta)

# COMMAND ----------

# Enable Time Travel (Versioned Data)
# Delta Lake automatically versions your data, allowing you to query older versions

# Check Table Versions

df_delta = spark.read.format("delta").option("versionAsOf", 0).load("/mnt/delta/sales_data")
display(df_delta)

# COMMAND ----------

# Convert all the Tables to Delta

# Clean the column names for all tables before saving them to Delta

# Function to clean column names
def clean_column_names(df):
    for col_name in df.columns:
        new_col_name = (
            col_name.replace(" ", "_")
            .replace(",", "")
            .replace(";", "")
            .replace("{", "")
            .replace("}", "")
            .replace("(", "")
            .replace(")", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("=", "")
        )
        df = df.withColumnRenamed(col_name, new_col_name)
    return df

# Clean all DataFrames
df_products = clean_column_names(df_products)
df_sales = clean_column_names(df_sales)
df_final = clean_column_names(df_final)

# COMMAND ----------

# Now save as Delta format

df_products.write.format("delta").mode("overwrite").save("/mnt/delta/products_data")
df_sales.write.format("delta").mode("overwrite").save("/mnt/delta/sales_data")
df_final.write.format("delta").mode("overwrite").save("/mnt/delta/final_sales_data")


# COMMAND ----------

# Visualize Total Sales per Month using Matplotlib in Databricks

# Convert Spark DataFrame to Pandas

import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas
pdf = df_monthly_sales.toPandas()

# Sort the data by "Order Month"
pdf = pdf.sort_values("Order Month")

# Plot the data
plt.figure(figsize=(10,5))
plt.plot(pdf["Order Month"], pdf["Total Sales by Month"], marker="o", linestyle="-", color="royalblue")

# Labels and title
plt.xlabel("Order Month")
plt.ylabel("Total Sales")
plt.title("Total Sales Per Month")
plt.xticks(rotation=45)

# Show the plot
plt.show()

# COMMAND ----------

# Bar chart for Total Sales by Product

pdf = df_product_sales.toPandas()
pdf = pdf.sort_values("Total Sales by Product", ascending=False)

plt.figure(figsize=(12,6))
plt.bar(pdf["Product Name"], pdf["Total Sales by Product"], color="royalblue")

plt.xlabel("Product Name")
plt.ylabel("Total Sales")
plt.title("Total Sales by Product")
plt.xticks(rotation=45)

plt.show()
