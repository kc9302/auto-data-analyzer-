"""
Mock SQLite Database Generator for Testing Auto Data Analyzer
Creates a realistic relational database with missing values, PII, outliers, and FK relations.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

def generate_mock_db(db_path: str = "tests/data/sample_warehouse.db", n_rows: int = 5000):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    np.random.seed(42)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. customer_churn table
    tenure = np.random.exponential(scale=20, size=n_rows).astype(int)
    tenure = np.clip(tenure, 0, 72)

    monthly_charges = np.random.uniform(18.5, 118.75, size=n_rows).round(2)
    
    # Total charges: roughly tenure * monthly_charges + noise
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=n_rows)
    total_charges = np.where(total_charges < 0, 0, total_charges).round(2)

    # Inject realistic missing values (12.5% missing in total_charges)
    missing_mask = (tenure == 0) | (np.random.rand(n_rows) < 0.10)
    total_charges_series = pd.Series(total_charges)
    total_charges_series[missing_mask] = np.nan

    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    payment_methods = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows)

    # Churn probability based on contract and tenure
    churn_prob = 0.2 + (contracts == "Month-to-month") * 0.3 - (tenure / 100) * 0.3
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = (np.random.rand(n_rows) < churn_prob).astype(int)

    # Generate PII
    rrn_list = [f"{np.random.randint(70, 99):02d}{np.random.randint(1, 13):02d}{np.random.randint(1, 29):02d}-{np.random.randint(1, 5)}{np.random.randint(100000, 999999)}" for _ in range(n_rows)]
    email_list = [f"user_{i}@example.com" for i in range(n_rows)]

    customers_df = pd.DataFrame({
        "customer_id": range(10001, 10001 + n_rows),
        "resident_id": rrn_list,                  # PII column
        "email": email_list,                      # PII column
        "gender": np.random.choice(["Male", "Female"], size=n_rows),
        "senior_citizen": np.random.choice([0, 1], size=n_rows, p=[0.84, 0.16]),
        "partner": np.random.choice(["Yes", "No"], size=n_rows),
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges_series,
        "contract": contracts,
        "payment_method": payment_methods,
        "churn": churn                            # Target variable
    })
    customers_df.to_sql("customers", conn, if_exists="replace", index=False)

    # 2. orders table (with Foreign Key relation to customers)
    n_orders = n_rows * 2
    orders_df = pd.DataFrame({
        "order_id": range(500001, 500001 + n_orders),
        "customer_id": np.random.choice(customers_df["customer_id"], size=n_orders),
        "amount": np.random.exponential(scale=45, size=n_orders).round(2),
        "order_status": np.random.choice(["COMPLETED", "SHIPPED", "CANCELLED"], size=n_orders, p=[0.8, 0.15, 0.05])
    })
    orders_df.to_sql("orders", conn, if_exists="replace", index=False)

    # Create an index
    cursor.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    conn.commit()
    conn.close()
    print(f"Successfully generated mock SQLite database at: {db_path}")

if __name__ == "__main__":
    generate_mock_db()
