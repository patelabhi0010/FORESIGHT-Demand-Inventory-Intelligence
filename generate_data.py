import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================
# 1. GENERATE SKU MASTER DATA
# ============================================
def generate_sku_master(n_skus=200):
    """Generate SKU master data with realistic product categories"""
    
    categories = {
        'Furniture': ['Sofas', 'Tables', 'Chairs', 'Storage Units', 'Bookshelves'],
        'Decor': ['Lighting', 'Wall Art', 'Vases', 'Mirrors', 'Clocks'],
        'Kitchen': ['Cookware', 'Utensils', 'Storage Containers', 'Small Appliances', 'Tableware'],
        'Bedding': ['Sheets', 'Comforters', 'Pillows', 'Blankets', 'Mattress Toppers'],
        'Bath': ['Towels', 'Shower Curtains', 'Bath Mats', 'Accessories', 'Robes'],
        'Lighting': ['Floor Lamps', 'Table Lamps', 'Ceiling Lights', 'String Lights'],
        'Storage': ['Baskets', 'Boxes', 'Shelves', 'Cabinets'],
        'Rugs': ['Area Rugs', 'Runner Rugs', 'Doormats']
    }
    
    sku_data = []
    
    for i in range(n_skus):
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        
        
        days_ago = random.randint(365, 1095)  # Launch date between 1-3 years ago (some newer, some older)
        launch_date = datetime.now() - timedelta(days=days_ago)
        
        # Different pricing tiers
        price_tier = random.choice(['budget', 'mid', 'premium'])
        if price_tier == 'budget':
            cost = random.uniform(50, 500)
            price = cost * random.uniform(1.4, 1.8)
        elif price_tier == 'mid':
            cost = random.uniform(500, 2000)
            price = cost * random.uniform(1.6, 2.0)
        else:  # premium
            cost = random.uniform(2000, 8000)
            price = cost * random.uniform(1.8, 2.5)
        
        sku_data.append({
            'sku_id': f'SKU{str(i+1).zfill(4)}',
            'category': category,
            'subcategory': subcategory,
            'launch_date': launch_date.strftime('%Y-%m-%d'),
            'unit_cost': round(cost, 2),
            'list_price': round(price, 2)
        })
    
    return pd.DataFrame(sku_data)

# ============================================
# 2. GENERATE CALENDAR DATA
# ============================================
def generate_calendar(start_date='2022-01-01', end_date='2024-12-31'):
    """Generate calendar dimension with Indian holidays and events"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Indian holidays (2022-2024)
    holidays = {
        '2022-01-26': 'Republic Day',
        '2022-03-18': 'Holi',
        '2022-04-15': 'Good Friday',
        '2022-04-14': 'Ambedkar Jayanti',
        '2022-05-03': 'Eid-ul-Fitr',
        '2022-05-01': 'Labour Day',
        '2022-08-15': 'Independence Day',
        '2022-08-19': 'Janmashtami',
        '2022-10-02': 'Gandhi Jayanti',
        '2022-10-05': 'Dussehra',
        '2022-10-24': 'Diwali',
        '2022-11-08': 'Guru Nanak Jayanti',
        '2022-12-25': 'Christmas',
        '2023-01-26': 'Republic Day',
        '2023-03-08': 'Holi',
        '2023-04-07': 'Good Friday',
        '2023-04-14': 'Ambedkar Jayanti',
        '2023-04-22': 'Eid-ul-Fitr',
        '2023-05-01': 'Labour Day',
        '2023-08-15': 'Independence Day',
        '2023-09-07': 'Janmashtami',
        '2023-10-02': 'Gandhi Jayanti',
        '2023-10-24': 'Dussehra',
        '2023-11-12': 'Diwali',
        '2023-11-27': 'Guru Nanak Jayanti',
        '2023-12-25': 'Christmas',
        '2024-01-26': 'Republic Day',
        '2024-03-25': 'Holi',
        '2024-03-29': 'Good Friday',
        '2024-04-11': 'Eid-ul-Fitr',
        '2024-05-01': 'Labour Day',
        '2024-08-15': 'Independence Day',
        '2024-10-02': 'Gandhi Jayanti',
        '2024-10-31': 'Diwali',
        '2024-12-25': 'Christmas'
    }
    
    promo_events = [
        ('2022-01-01', '2022-01-10', 'New Year Sale'),
        ('2022-02-10', '2022-02-15', 'Valentine Sale'),
        ('2022-06-01', '2022-06-15', 'Summer Sale'),
        ('2022-08-10', '2022-08-20', 'Independence Day Sale'),
        ('2022-10-20', '2022-11-05', 'Diwali Festival'),
        ('2022-12-15', '2022-12-31', 'Christmas Sale'),
        ('2023-01-01', '2023-01-10', 'New Year Sale'),
        ('2023-06-01', '2023-06-15', 'Summer Sale'),
        ('2023-08-10', '2023-08-20', 'Independence Day Sale'),
        ('2023-10-20', '2023-11-05', 'Diwali Festival'),
        ('2023-11-20', '2023-11-30', 'Black Friday Sale'),
        ('2023-12-15', '2023-12-31', 'Christmas Sale'),
        ('2024-01-01', '2024-01-10', 'New Year Sale'),
        ('2024-06-01', '2024-06-15', 'Summer Sale'),
        ('2024-08-10', '2024-08-20', 'Independence Day Sale'),
        ('2024-10-20', '2024-11-05', 'Diwali Festival'),
    ]
    
    calendar_data = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Season logic
        month = date.month
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        else:
            season = 'Fall'
        
        # Check if holiday
        is_holiday = 1 if date_str in holidays else 0
        holiday_name = holidays.get(date_str, None)
        
        # Check if promotion event
        promo_event = None
        for start, end, event in promo_events:
            if start <= date_str <= end:
                promo_event = event
                break
        
        calendar_data.append({
            'date': date_str,
            'year': date.year,
            'week': date.isocalendar()[1],
            'month': month,
            'quarter': (month - 1) // 3 + 1,
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'season': season,
            'is_holiday': is_holiday,
            'holiday_name': holiday_name,
            'promo_event': promo_event
        })
    
    return pd.DataFrame(calendar_data)

# ============================================
# 3. GENERATE INVENTORY SNAPSHOTS
# ============================================
def generate_inventory_snapshots(sku_df, start_date='2022-01-01', end_date='2024-12-31', frequency='W'):
    """Generate weekly inventory snapshots with realistic patterns"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    inventory_data = []
    
    for sku in sku_df.itertuples():
        # Base demand characteristics for this SKU
        base_demand = random.uniform(2, 60)  # daily average
        demand_variability = random.uniform(0.2, 0.5)
        
        # Seasonality pattern for this SKU
        seasonal_amplitude = random.uniform(0.1, 0.4)
        seasonal_peak_month = random.randint(1, 12)
        
        # Lead time based on product type
        if sku.category in ['Furniture', 'Rugs']:
            lead_time = random.randint(14, 30)
        elif sku.category in ['Kitchen', 'Lighting']:
            lead_time = random.randint(7, 21)
        else:
            lead_time = random.randint(3, 14)
        
        # Reorder policy
        safety_stock_factor = random.uniform(1.5, 3.0)
        reorder_point = int(base_demand * lead_time * safety_stock_factor)
        
        for date in dates:
            # Calculate seasonal factor for this date
            month = date.month
            # Sinusoidal seasonal pattern
            seasonal_factor = 1 + seasonal_amplitude * np.sin(
                2 * np.pi * (month - seasonal_peak_month) / 12
            )
            
            # Current expected daily demand
            current_demand = base_demand * seasonal_factor
            
            # Stock level with realistic variation
            # Higher stock before peak season, lower after
            base_stock = current_demand * lead_time * random.uniform(2, 4)
            
            # Add random variation
            on_hand = max(0, int(base_stock * (1 + np.random.normal(0, demand_variability))))
            
            # Sometimes we have on-order stock (especially if below reorder point)
            if on_hand < reorder_point:
                on_order = int(random.uniform(0.5, 2.0) * reorder_point)
            else:
                on_order = int(random.uniform(0, 0.3) * reorder_point) if random.random() < 0.3 else 0
            
            inventory_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'sku_id': sku.sku_id,
                'on_hand_units': max(0, on_hand),
                'on_order_units': max(0, on_order),
                'lead_time_days': lead_time,
                'reorder_point': reorder_point
            })
    
    return pd.DataFrame(inventory_data)

# ============================================
# 4. GENERATE SALES DATA
# ============================================
def generate_sales_data(sku_df, calendar_df, start_date='2022-01-01', end_date='2024-12-31'):
    """Generate daily sales data with realistic patterns"""
    
    sales_data = []
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Convert calendar to dict for quick lookup
    calendar_dict = {}
    for row in calendar_df.itertuples():
        calendar_dict[row.date] = {
            'is_holiday': row.is_holiday,
            'promo_event': row.promo_event,
            'is_weekend': row.is_weekend,
            'season': row.season,
            'month': row.month
        }
    
    for sku in sku_df.itertuples():
        # Base sales characteristics
        base_avg = random.uniform(1, 40)
        
        # SKU-specific patterns
        weekday_multiplier = random.uniform(0.8, 1.2)
        weekend_multiplier = random.uniform(1.1, 1.8)
        
        # Seasonality
        seasonal_amplitude = random.uniform(0.1, 0.5)
        peak_month = random.randint(1, 12)
        
        # Promotion sensitivity
        promo_boost = random.uniform(1.5, 3.0)
        
        # Price elasticity (how much sales drop with price changes)
        price_elasticity = random.uniform(0.5, 1.5)
        
        # Dead stock probability (some SKUs just don't sell well)
        is_dead_stock = random.random() < 0.05  # 5% are dead stock
        
        # Trend (growing or declining)
        trend = random.uniform(-0.3, 0.5)  # negative = declining, positive = growing
        launch_date = datetime.strptime(sku.launch_date, '%Y-%m-%d')
        
        for date in date_list:
            date_str = date.strftime('%Y-%m-%d')
            days_since_launch = (date - launch_date).days
            
            # Don't sell before launch
            if days_since_launch < 0:
                continue
            
            # Get calendar data
            cal_info = calendar_dict.get(date_str, {})
            is_holiday = cal_info.get('is_holiday', 0)
            promo_event = cal_info.get('promo_event', None)
            is_weekend = cal_info.get('is_weekend', 0)
            month = cal_info.get('month', date.month)
            
            # Base daily sales with trend
            days_factor = min(1.0, days_since_launch / 30)  # Ramp up in first month
            trend_factor = 1 + (days_since_launch / 365) * trend
            
            if is_dead_stock:
                daily_sales = base_avg * 0.1 * random.uniform(0.5, 1.5)
            else:
                daily_sales = base_avg * days_factor * trend_factor
            
            # Weekday effect
            if is_weekend:
                daily_sales *= weekend_multiplier
            else:
                daily_sales *= weekday_multiplier
            
            # Seasonality
            seasonal_factor = 1 + seasonal_amplitude * np.sin(
                2 * np.pi * (month - peak_month) / 12
            )
            daily_sales *= seasonal_factor
            
            # Holiday boost
            if is_holiday:
                daily_sales *= random.uniform(1.3, 2.0)
            
            # Promotion boost
            if promo_event:
                daily_sales *= random.uniform(1.5, promo_boost)
                # During promotion, price might be discounted
                price_discount = random.uniform(0.7, 0.95)
            else:
                price_discount = 1.0
            
            # Add random noise
            noise = np.random.normal(0, daily_sales * 0.2)
            daily_sales = max(0, daily_sales + noise)
            
            # Round to integer
            units_sold = int(round(daily_sales))
            
            # Skip zeros occasionally (out of stock days)
            if units_sold == 0 and random.random() < 0.5:
                continue
            
            # Price with variation
            base_price = sku.list_price * price_discount
            price = base_price * random.uniform(0.95, 1.05)
            price = round(price, 2)
            
            revenue = units_sold * price
            
            # Promo flag
            promo_flag = 1 if promo_event else 0
            
            sales_data.append({
                'date': date_str,
                'sku_id': sku.sku_id,
                'units_sold': units_sold,
                'revenue': round(revenue, 2),
                'unit_price': price,
                'promo_flag': promo_flag
            })
    
    return pd.DataFrame(sales_data)

# ============================================
# 5. ADD DATA IMPERFECTIONS (Realistic issues)
# ============================================
def add_imperfections(df, df_type='sales'):
    """Add realistic data quality issues"""
    
    df = df.copy()
    
    if df_type == 'sales':
        # Add some missing values
        missing_mask = np.random.random(len(df)) < 0.005
        df.loc[missing_mask, 'units_sold'] = np.nan
        
        # Add some duplicate rows (rare)
        if len(df) > 1000:
            dup_indices = np.random.choice(len(df), size=int(len(df)*0.001), replace=False)
            dups = df.iloc[dup_indices].copy()
            df = pd.concat([df, dups], ignore_index=True)
    
    elif df_type == 'sku':
        # Add some missing categories (rare)
        missing_mask = np.random.random(len(df)) < 0.01
        df.loc[missing_mask, 'category'] = np.nan
        
        # Add some inconsistent formatting
        df.loc[np.random.random(len(df)) < 0.005, 'sku_id'] = df.loc[
            np.random.random(len(df)) < 0.005, 'sku_id'
        ].str.lower()
    
    elif df_type == 'inventory':
        # Add negative stock (rare, data entry errors)
        neg_mask = np.random.random(len(df)) < 0.001
        df.loc[neg_mask, 'on_hand_units'] = -abs(df.loc[neg_mask, 'on_hand_units'])
        
        # Add unusually large orders
        large_mask = np.random.random(len(df)) < 0.001
        df.loc[large_mask, 'on_order_units'] = df.loc[large_mask, 'on_order_units'] * 10
    
    return df

# ============================================
# 6. MAIN EXECUTION
# ============================================
def generate_all_data():
    """Generate all CSV files with realistic data"""
    
    print("=" * 60)
    print(" NorthBay Living - Data Generator")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    
    print("\n Generating SKU Master data...")
    sku_df = generate_sku_master(200)
    sku_df = add_imperfections(sku_df, 'sku')
    sku_df.to_csv('data/raw/sku_master.csv', index=False)
    print(f"   : {len(sku_df)} SKUs created")
    print(f"   : Categories: {sku_df['category'].nunique()}")
    print(f"   : Price range: ₹{sku_df['list_price'].min():.0f} - ₹{sku_df['list_price'].max():.0f}")
    
    print("\n Generating Calendar data...")
    calendar_df = generate_calendar('2022-01-01', '2024-12-31')
    calendar_df.to_csv('data/raw/calendar.csv', index=False)
    print(f"   : {len(calendar_df)} days created")
    print(f"   : {calendar_df['is_holiday'].sum()} holidays included")
    print(f"   : {calendar_df['promo_event'].notna().sum()} promotion days")
    
    print("\n Generating Inventory Snapshots...")
    inventory_df = generate_inventory_snapshots(sku_df, '2022-01-01', '2024-12-31')
    inventory_df = add_imperfections(inventory_df, 'inventory')
    inventory_df.to_csv('data/raw/inventory_snapshots.csv', index=False)
    print(f"   : {len(inventory_df)} inventory records created")
    print(f"   : Average on-hand: {inventory_df['on_hand_units'].mean():.0f} units")
    
    print("\n Generating Sales Data...")
    sales_df = generate_sales_data(sku_df, calendar_df, '2022-01-01', '2024-12-31')
    sales_df = add_imperfections(sales_df, 'sales')
    sales_df.to_csv('data/raw/sales_daily.csv', index=False)
    print(f"    {len(sales_df)} sales records created")
    print(f"    Total revenue: ₹{sales_df['revenue'].sum():.2f}")
    print(f"    Average daily sales: {sales_df['units_sold'].mean():.1f} units")
    
    print("\n" + "=" * 60)
    print(" ALL CSV FILES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n File locations:")
    print("    data/raw/sku_master.csv")
    print("    data/raw/calendar.csv")
    print("    data/raw/inventory_snapshots.csv")
    print("    data/raw/sales_daily.csv")
    
    print("\n Quick Data Preview:")
    print("\n SKU Master (first 3 rows):")
    print(sku_df.head(3).to_string())
    
    print("\n Calendar (first 3 rows):")
    print(calendar_df.head(3).to_string())
    
    print("\n Inventory (first 3 rows):")
    print(inventory_df.head(3).to_string())
    
    print("\n Sales (first 3 rows):")
    print(sales_df.head(3).to_string())
    
    print("\n" + "=" * 60)
    print(" Data Summary Statistics:")
    print("=" * 60)
    print(f"\nTotal Sales: {len(sales_df):,} records")
    print(f"Total Revenue: ₹{sales_df['revenue'].sum():,.2f}")
    print(f"Total Units Sold: {sales_df['units_sold'].sum():,}")
    print(f"Average Price: ₹{sales_df['unit_price'].mean():.2f}")
    print(f"Promotion Sales: {sales_df['promo_flag'].sum():,} records ({sales_df['promo_flag'].mean()*100:.1f}%)")
    
    print("\n Sales by Category:")
    sales_with_cat = sales_df.merge(
        sku_df[['sku_id', 'category']], 
        on='sku_id', 
        how='left'
    )
    category_sales = sales_with_cat.groupby('category')['revenue'].sum().sort_values(ascending=False)
    for cat, revenue in category_sales.head(5).items():
        print(f"   {cat}: ₹{revenue:,.2f}")

# Run the generator
if __name__ == "__main__":
    generate_all_data()