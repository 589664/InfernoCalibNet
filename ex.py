from src.utils.DatasetTools import preprocess_and_split_csv
from rich import print

stats = preprocess_and_split_csv()
print(stats)
