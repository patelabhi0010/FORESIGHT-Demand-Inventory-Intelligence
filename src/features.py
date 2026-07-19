"""
Feature Engineering Module (UPDATED)
Includes improved features based on feedback
"""

import pandas as pd
import numpy as np

def load_processed_data(filepath='data/processed/final_dataset.csv'):
    """Load the cleaned dataset"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    return df

def create_date_features(df):
    """Extract date-based features (with cyclic encoding)"""
    df = df.copy()
    
    # Basic date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclic features for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def create_lag_features(df, lags=[1, 7, 14, 28,56]):
    """Create lag features (past sales) - Added lag 14 and 28"""
    df = df.copy()
    df = df.sort_values(['sku_id', 'date'])
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku_id')['units_sold'].shift(lag)
    
    return df

def create_rolling_features(df, windows=[7, 14, 28,32]):
    """
    Create rolling statistics (SHIFTED to avoid data leakage)
    The shift(1) ensures we don't use current day's data
    """
    df = df.copy()
    df = df.sort_values(['sku_id', 'date'])
    
    for window in windows:
        # Rolling mean (shifted by 1 to avoid leakage)
        df[f'rolling_mean_{window}'] = df.groupby('sku_id')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation (shifted by 1)
        df[f'rolling_std_{window}'] = df.groupby('sku_id')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    return df

def create_inventory_features(df):
    """Create features from inventory data"""
    df = df.copy()
    
    # Stockout indicator
    df['is_below_reorder_point'] = (df['on_hand_units'] < df['reorder_point']).astype(int)
    
    # Stock gap
    df['stock_gap'] = df['on_hand_units'] - df['reorder_point']
    
    # Inventory coverage using rolling mean instead of current sales
    df['inventory_coverage'] = df['on_hand_units'] / (df['rolling_mean_7'] + 1)
    
    return df

def create_holiday_features(df):
    """Create holiday and promotion features"""
    df = df.copy()
    
    # Holiday indicator
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    
    # Promotion indicator
    df['has_promo'] = df['promo_flag'].fillna(0).astype(int)
    
    return df

def create_price_features(df):
    """Create price-related features"""
    df = df.copy()
    
    # Price tier
    df['price_tier'] = pd.cut(
        df['list_price'],
        bins=[0, 500, 2000, 10000, float('inf')],
        labels=['budget', 'mid', 'premium', 'luxury']
    )
    
    # Discount percentage
    df['discount'] = ((df['list_price'] - df['unit_price']) / df['list_price']).fillna(0)
    
    return df

def create_sales_features(df):
    """Create sales-based features"""
    df = df.copy()
    df = df.sort_values(['sku_id', 'date'])
    
    # Sales change
    # FIX: shift(1) BEFORE pct_change. The original
    # `pct_change(periods=7)` compares units_sold[t] to units_sold[t-7],
    # which uses row t's own target value inside the feature for row t -
    # that is target leakage (units_sold[t] can be algebraically solved
    # back out of the feature). Shifting first makes it purely historical,
    # consistent with how lag_/rolling_ features already avoid this.
    df['sales_change_7d'] = df.groupby('sku_id')['units_sold'].transform(
        lambda x: x.shift(1).pct_change(periods=7)
    )
    
    # Days since launch
    df['days_since_launch'] = (df['date'] - df['launch_date']).dt.days
    
    return df

def engineer_features(df):
    """Master function to create all features"""
    print(" Starting feature engineering...")
    
    print("  → Adding date features (with cyclic encoding)...")
    df = create_date_features(df)
    
    print("  → Creating lag features (1,7,14,28,56)...")
    df = create_lag_features(df)
    
    print("  → Creating rolling statistics (shifted for no leakage)...")
    df = create_rolling_features(df)
    
    print("  → Adding inventory features...")
    df = create_inventory_features(df)
    
    print("  → Adding holiday features...")
    df = create_holiday_features(df)
    
    print("  → Adding price features...")
    df = create_price_features(df)
    
    print("  → Adding sales features...")
    df = create_sales_features(df)
    
    print(" Feature engineering complete!")
    print(f"  → Shape: {df.shape}")
    
    return df

def prepare_features_for_modeling(df):
    """Prepare final dataset for modeling"""
    df = df.copy()
    
    print(" Preparing features for modeling...")
    
    # =============================================
    # KEEP essential identifiers for tracking
    # =============================================
    # Keep these for grouping, filtering, and output
    keep_columns = ['date', 'sku_id', 'category', 'subcategory']
    
    # =============================================
    # DROP only columns that cause issues
    # =============================================
    columns_to_drop = [
       
        'holiday_name', 'promo_event', 'season',
       
    ]
    
    # =============================================
    # Separate target from features
    # =============================================
    # Keep 'units_sold' as target
    # Don't drop it!
    
    # Drop only unnecessary columns
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # =============================================
    # Handle missing values
    # =============================================
    print("  → Handling missing values...")
    
    # Numeric columns (excluding identifiers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Don't fill missing for identifiers
    numeric_cols = [col for col in numeric_cols if col not in ['sku_id']]
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns (including sku_id)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # One-hot encode categorical variables (but keep sku_id!)
    print("  → Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Don't one-hot encode sku_id - keep it as identifier
    if 'sku_id' in categorical_cols:
        categorical_cols = [col for col in categorical_cols if col != 'sku_id']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(" Feature preparation complete!")
    print(f"  → Final shape: {df.shape}")
    print(f"  → Columns: {df.columns.tolist()[:10]}...")
    
    return df


def save_features(df, filepath='data/processed/featured_dataset.csv'):
    """Save the feature-engineered dataset"""
    df.to_csv(filepath, index=False)
    print(f" Features saved to {filepath}")

if __name__ == "__main__":
  
    
   
    df = load_processed_data()
    print(f"\n Loaded {len(df)} rows")
    
    df = engineer_features(df)
    df = prepare_features_for_modeling(df)
    save_features(df)
    
   
    print(" FEATURE ENGINEERING COMPLETE!")
   