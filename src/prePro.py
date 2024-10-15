import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_metadata(csv_path: str, save_path: str, num_rows: int = 14999) -> pd.DataFrame:
    """
    Preprocesses metadata from a CSV file and saves the result to another CSV file.

    Args:
        csv_path (str): The path to the input CSV file containing metadata.
        save_path (str): The path where the preprocessed metadata will be saved.
        num_rows (int, optional): The number of rows to limit the data to. Defaults to 14999.

    Returns:
        pd.DataFrame: The preprocessed metadata as a pandas DataFrame.
    """
    # Load and filter metadata
    df = pd.read_csv(csv_path)
    filtered_df = df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender', 'View Position']]
    filtered_df.columns = ['ImageID', 'Labels', 'Age', 'Gender', 'XrayView']

    # Limit the rows
    filtered_df = filtered_df.head(num_rows)

    # Clean the ImageID column
    filtered_df.loc[:, 'ImageID'] = filtered_df['ImageID'].str.replace('.png', '', regex=False)

    # Split Labels and binarize them
    filtered_df['Labels'] = filtered_df['Labels'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    filtered_df['MultiHotLabels'] = mlb.fit_transform(filtered_df['Labels']).tolist()

    # Convert Gender and XrayView to numerical values
    filtered_df['Gender'] = filtered_df['Gender'].map({'M': 0, 'F': 1})
    filtered_df['XrayView'] = filtered_df['XrayView'].map({'PA': 0, 'AP': 1})

    # Normalize Age
    filtered_df['Age'] = (filtered_df['Age'] - filtered_df['Age'].mean()) / filtered_df['Age'].std()

    # Save preprocessed metadata
    filtered_df.to_csv(save_path, index=False)

    return filtered_df
