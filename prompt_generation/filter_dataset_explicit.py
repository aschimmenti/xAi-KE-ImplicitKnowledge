import pandas as pd

# Read both CSV files
output_prompts = pd.read_csv('output_prompts-v2.csv')
wikidata_statements = pd.read_csv('wikidata_statements.csv')

# Get unique IDs from output_prompts
output_ids = output_prompts['id'].unique()

# Filter wikidata statements to only include rows where id is in output_ids
filtered_statements = wikidata_statements[wikidata_statements['id'].isin(output_ids)]

# Create a dictionary mapping entity IDs to their selected properties and values
selected_props = {}
for _, row in output_prompts.iterrows():
    selected_props[row['id']] = (row['selected_property'], row['property_value'])

# Add new column indicating if this is the selected property
def is_selected_property(row):
    if row['id'] in selected_props:
        selected_property, selected_value = selected_props[row['id']]
        return (row['predicate'] == selected_property) and (row['object'] == selected_value)
    return False

filtered_statements['is_selected_property'] = filtered_statements.apply(is_selected_property, axis=1)

# Save the filtered dataframe to a new CSV
filtered_statements.to_csv('filtered_wikidata_statements.csv', index=False)

# Print some statistics
print(f"Number of unique IDs in output_prompts: {len(output_ids)}")
print(f"Number of rows in original wikidata_statements: {len(wikidata_statements)}")
print(f"Number of rows in filtered statements: {len(filtered_statements)}")
print("\nSample of filtered statements with selected property marking:")
print(filtered_statements[['id', 'predicate', 'object', 'is_selected_property']].head(10))