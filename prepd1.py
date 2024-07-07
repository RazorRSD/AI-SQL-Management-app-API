import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('synthetic_text_to_sql_train.csv')

# Function to extract table schema from sql_context
def extract_schema(sql_context):
    create_table = sql_context.split(';')[0]
    table_name = create_table.split('(')[0].split()[-1]
    columns = create_table.split('(')[1].split(')')[0].split(',')
    columns = [col.strip().split()[0] for col in columns]
    return f"{table_name}({', '.join(columns)})"

# Prepare the data
data = []
for _, row in df.iterrows():
    data.append({
        'question': row['sql_prompt'],
        'schema': extract_schema(row['sql_context']),
        'query': row['sql']
    })

# Convert to DataFrame
prepared_df = pd.DataFrame(data)

# Split the data into train and validation sets
train_df, val_df = train_test_split(prepared_df, test_size=0.2, random_state=42)

# Save to JSON files (as datasets library works well with JSON)
train_df.to_json('train_data.json', orient='records', lines=True)
val_df.to_json('val_data.json', orient='records', lines=True)

print(f"Saved {len(train_df)} training examples to train_data.json")
print(f"Saved {len(val_df)} validation examples to val_data.json")

# Display a sample of the prepared data
print("\nSample of prepared data:")
print(prepared_df.head().to_string())

# Optionally, save to CSV as well
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
print("\nAlso saved data to train_data.csv and val_data.csv")