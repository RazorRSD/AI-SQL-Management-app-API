from datasets import load_dataset
import pandas as pd
import os

# Load the dataset
ds = load_dataset("gretelai/synthetic_text_to_sql")

# The dataset might have multiple splits (e.g., 'train', 'test', 'validation')
# Let's convert each split to a separate CSV file

for split in ds.keys():
    # Convert the split to a pandas DataFrame
    df = pd.DataFrame(ds[split])
    
    # Create a filename for the CSV
    filename = f"synthetic_text_to_sql_{split}.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    
    print(f"Saved {split} split to {filename}")

# Print out the first few rows of the first split (usually 'train')
first_split = list(ds.keys())[0]
print(f"\nFirst few rows of the {first_split} split:")
print(df.head())

# Print out the column names
print("\nColumn names:")
print(df.columns.tolist())

# Print out the number of rows in each split
for split in ds.keys():
    print(f"\nNumber of rows in {split} split: {len(ds[split])}")