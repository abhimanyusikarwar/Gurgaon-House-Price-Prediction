import pandas as pd

pd.read_csv("input.csv").drop(columns="median_house_value").to_csv("input.csv", index=False)