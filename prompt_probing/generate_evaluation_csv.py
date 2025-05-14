import pandas as pd

def merge_datasets(file1_path, file2_path, output_path):
    """
    Merge two datasets based on ID and add selected_property and property_value columns
    from the first dataset to the second dataset.
    
    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file containing prompt and property information
    file2_path : str
        Path to the second CSV file containing entity information
    output_path : str
        Path where the merged CSV file will be saved
    """
    
    # Read both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Create a mapping dictionary from df1
    property_mapping = df1.set_index('id')[['selected_property', 'property_value']]
    
    # Merge the datasets
    result = df2.merge(
        property_mapping,
        left_on='id',
        right_index=True,
        how='left',
        validate='1:1'
    )
    
    # Verify the merge
    print(f"Original number of rows in second dataset: {len(df2)}")
    print(f"Number of rows after merge: {len(result)}")
    print(f"Number of rows with matched properties: {result['selected_property'].notna().sum()}")
    
    # Save the result
    result.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to {output_path}")
    
    # Display a sample of the merged data
    print("\nSample of merged data:")
    print(result[['id', 'entity_name', 'selected_property', 'property_value']].head())

# Usage example:
if __name__ == "__main__":
    file1 = "explicit_qa_dataset.csv"  # File with prompts and properties
    file2 = "explicit_qa_results-v2.csv"  # File with entity information
    output = "explicit_merged_dataset-v2.csv"
    
    merge_datasets(file1, file2, output)