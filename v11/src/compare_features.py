import pandas as pd
import numpy as np
import os

FEATURES_PATH = r'e:\crop-disease-outbreak-prediction-system-feature-zip-changes\crop-disease-outbreak-prediction-system-feature-zip-changes\model_train\v11\data\processed\v11_features.csv'
df = pd.read_csv(FEATURES_PATH)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Compare key features between val years and test years, monsoon season only
monsoon = df[df['date'].dt.month.isin([6,7,8,9,10,11])]

val_years  = monsoon[monsoon['year'].isin([2015,2016,2017,2018])]
test_years = monsoon[monsoon['year'].isin([2019,2020,2021])]

features = ['RH2M', 'T2M', 'PRECTOTCORR', 'RH_persist_7d', 'Rain_sum_14d', 'RH2M_latent_window', 'T2M_latent_window']

print('Feature means: Val (2015-18) vs Test (2019-21), monsoon season')
print(f'{"Feature":<22} {"Val mean":>10} {"Test mean":>10} {"Ratio":>8}')
print('-' * 55)
for f in features:
    v = val_years[f].mean()
    t = test_years[f].mean()
    ratio = t/v if v != 0 else float('nan')
    print(f'{f:<22} {v:>10.3f} {t:>10.3f} {ratio:>8.2f}')
