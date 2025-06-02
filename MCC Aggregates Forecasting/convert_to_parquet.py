# %% [markdown]
# # Convert CSV to Parquet Format
# Utility to convert large CSV files to Parquet for faster I/O

# %%
import pandas as pd
import os
import sys
from datetime import datetime

print("CSV to Parquet Converter")

# %% Environment setup
def detect_environment():
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive/')
        return 'colab', '/content/drive/MyDrive/fcst'
    except ImportError:
        return 'local', '..'

environment, base_path = detect_environment()
print(f"Environment: {environment}")
print(f"Base path: {base_path}")

# %% Convert transactions data
print("\n=== Converting transactions_data.csv ===")
csv_path = f'{base_path}/data/transactions_data.csv'
parquet_path = f'{base_path}/data/transactions_data.parquet'

if os.path.exists(csv_path) and not os.path.exists(parquet_path):
    print("Reading CSV file (this may take a while)...")
    start = datetime.now()
    
    # Read in chunks for memory efficiency
    chunk_size = 1_000_000
    chunks = []
    
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        chunks.append(chunk)
        print(f"  Processed {(i+1) * chunk_size:,} rows...", end='\r')
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal rows: {len(df):,}")
    
    # Convert data types to save space
    print("Optimizing data types...")
    df['amount'] = df['amount'].str.replace('$', '').astype('float32')
    df['client_id'] = df['client_id'].astype('int32')
    df['mcc'] = df['mcc'].astype('category')
    
    # Save as parquet
    print("Saving as parquet...")
    df.to_parquet(parquet_path, compression='snappy')
    
    # Compare file sizes
    csv_size = os.path.getsize(csv_path) / (1024**2)
    parquet_size = os.path.getsize(parquet_path) / (1024**2)
    
    print(f"\nConversion complete!")
    print(f"CSV size: {csv_size:.1f} MB")
    print(f"Parquet size: {parquet_size:.1f} MB")
    print(f"Compression ratio: {csv_size/parquet_size:.1f}x")
    print(f"Time taken: {(datetime.now() - start).total_seconds():.1f} seconds")
else:
    if os.path.exists(parquet_path):
        print("Parquet file already exists!")
    else:
        print("CSV file not found!")

# %% Convert other CSV files
print("\n=== Converting other CSV files ===")

csv_files = [
    'data/users_data.csv',
    'data/preprocessed/train_raw.csv',
    'data/preprocessed/test_raw.csv',
    'data/features/baseline_train.csv',
    'data/features/baseline_test.csv',
    'data/features/ml_train.csv',
    'data/features/ml_test.csv',
    'data/features/dl_train.csv',
    'data/features/dl_test.csv'
]

for csv_file in csv_files:
    csv_path = f'{base_path}/{csv_file}'
    parquet_path = csv_path.replace('.csv', '.parquet')
    
    if os.path.exists(csv_path) and not os.path.exists(parquet_path):
        print(f"\nConverting {csv_file}...")
        df = pd.read_csv(csv_path)
        
        # Optimize data types if possible
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        df.to_parquet(parquet_path, compression='snappy')
        
        csv_size = os.path.getsize(csv_path) / (1024**2)
        parquet_size = os.path.getsize(parquet_path) / (1024**2)
        print(f"  CSV: {csv_size:.1f} MB → Parquet: {parquet_size:.1f} MB ({csv_size/parquet_size:.1f}x compression)")

# %% Summary
print("\n=== CONVERSION SUMMARY ===")
print("Parquet advantages:")
print("- 5-10x faster to read/write")
print("- 2-5x smaller file size")
print("- Preserves data types (no parsing needed)")
print("- Column-oriented (efficient for analytics)")
print("\nTo use parquet files in your code:")
print("  df = pd.read_parquet('file.parquet') # instead of pd.read_csv('file.csv')")
print("\nThe optimized scripts automatically use parquet when available!") 