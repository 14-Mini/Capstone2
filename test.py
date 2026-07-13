import pandas as pd
df = pd.read_csv("full_labeled.csv")
print(df['attack_type'].value_counts(normalize=True) * 100)