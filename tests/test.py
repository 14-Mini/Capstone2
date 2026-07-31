import pandas as pd
df = pd.read_csv("data/splits/full_labeled.csv")
print(df['attack_type'].value_counts(normalize=True) * 100)