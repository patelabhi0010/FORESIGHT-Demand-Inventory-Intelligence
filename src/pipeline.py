import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data():
    """Load all raw CSV files"""
    sales = pd.read_csv('data/raw/sales_daily.csv')
    sku = pd.read_csv('data/raw/sku_master.csv')
    calendar = pd.read_csv('data/raw/calendar.csv')
    inventory = pd.read_csv('data/raw/inventory_snapshots.csv')
    return sales, sku, calendar, inventory

def clean_data(sales, sku, calendar, inventory):
    """Clean each dataset"""
    
    # 1. Fix SKU IDs that were lowercased (imperfection added)
    sku['sku_id'] = sku['sku_id'].str.upper()
    
    # 2. Convert dates
    sales['date'] = pd.to_datetime(sales['date'])
    calendar['date'] = pd.to_datetime(calendar['date'])
    inventory['date'] = pd.to_datetime(inventory['date'])
    sku['launch_date'] = pd.to_datetime(sku['launch_date'])
    
    # 3. Handle missing values in sales
    sales['units_sold'] = sales['units_sold'].fillna(0)
    sales['revenue'] = sales['revenue'].fillna(0)
    
    # 4. Handle missing categories in sku
    sku['category'] = sku['category'].fillna('Unknown')
    sku['subcategory'] = sku['subcategory'].fillna('Unknown')
    
    # 5. Remove duplicates
    sales = sales.drop_duplicates()
    sku = sku.drop_duplicates(subset=['sku_id'])
    calendar = calendar.drop_duplicates(subset=['date'])
    inventory = inventory.drop_duplicates()
    
    # 6. Fix negative inventory (imperfection added)
    inventory['on_hand_units'] = inventory['on_hand_units'].abs()
    inventory['on_order_units'] = inventory['on_order_units'].abs()
    
    return sales, sku, calendar, inventory

def merge_data(sales, sku, calendar, inventory):
    merged = sales.merge(sku, on='sku_id', how='left')
    merged = merged.merge(calendar, on='date', how='left')
    merged = merged.merge(inventory, on=['sku_id', 'date'], how='left')
    
    # Fill missing inventory values
    merged = merged.sort_values(['sku_id', 'date'])
    
    # Forward fill (carry forward)
    merged['on_hand_units'] = merged.groupby('sku_id')['on_hand_units'].fillna(method='ffill')
    merged['on_order_units'] = merged.groupby('sku_id')['on_order_units'].fillna(method='ffill')
    merged['lead_time_days'] = merged.groupby('sku_id')['lead_time_days'].fillna(method='ffill')
    merged['reorder_point'] = merged.groupby('sku_id')['reorder_point'].fillna(method='ffill')
    
    # Backward fill (carry backward for early dates)
    merged['on_hand_units'] = merged.groupby('sku_id')['on_hand_units'].fillna(method='bfill')
    merged['on_order_units'] = merged.groupby('sku_id')['on_order_units'].fillna(method='bfill')
    merged['lead_time_days'] = merged.groupby('sku_id')['lead_time_days'].fillna(method='bfill')
    merged['reorder_point'] = merged.groupby('sku_id')['reorder_point'].fillna(method='bfill')

    # Final handling if any NaN still exists
    merged['on_hand_units'] = merged['on_hand_units'].fillna(0)
    merged['on_order_units'] = merged['on_order_units'].fillna(0)

    merged['lead_time_days'] = merged['lead_time_days'].fillna(
    merged['lead_time_days'].median()
    )

    merged['reorder_point'] = merged['reorder_point'].fillna(
    merged['reorder_point'].median()
    )
   
    return merged

def save_data(df):
    """Save the final cleaned dataset"""
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/final_dataset.csv', index=False)
    print(f" Saved {len(df)} records to data/processed/final_dataset.csv")
    return df

def run_pipeline():
    """Run the entire pipeline"""
    print(" Starting data pipeline...")
    
    print("Loading data")
    sales, sku, calendar, inventory = load_data()
    
    print(" Cleaning data")
    sales, sku, calendar, inventory = clean_data(sales, sku, calendar, inventory)
    
    print("Merging data")
    merged = merge_data(sales, sku, calendar, inventory)
    
    print("Saving data")
    save_data(merged)
    
    print("Pipeline complete!")
    return merged

if __name__ == "__main__":
    run_pipeline()