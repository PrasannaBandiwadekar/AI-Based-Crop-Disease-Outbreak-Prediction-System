import pandas as pd
import os

GT_PATH = "research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv"

gt_df = pd.read_csv(GT_PATH)
print(gt_df)
print(f"Total outbreak events in GT: {len(gt_df)}")
gt_df['peak_start'] = pd.to_datetime(gt_df['peak_start'])
train_events = gt_df[gt_df['peak_start'].between('2005-01-01', '2018-12-31')]
print(f"Events in train window (2005-2018): {len(train_events)}")
print(train_events)
