import pandas as pd
signals_df = pd.read_csv('data/logs/signal_log.csv')
outcomes_df = pd.read_csv('data/reports/outcomes.csv')
signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], errors='coerce')
outcomes_df['timestamp'] = pd.to_datetime(outcomes_df['timestamp'], errors='coerce')
print("signals_df pairs:", signals_df['pair'].unique())
print("outcomes_df pairs:", outcomes_df['pair'].unique())
print("signals_df timestamps:", signals_df['timestamp'].head())
print("outcomes_df timestamps:", outcomes_df['timestamp'].head())
merged_df = signals_df.merge(outcomes_df[['pair', 'timestamp', 'outcome', 'pnl']], 
                           on=['pair', 'timestamp'], how='left')
print("Merged DataFrame:\n", merged_df[['pair', 'timestamp', 'outcome']].head())
print("Missing outcomes:\n", merged_df[merged_df['outcome'].isna()])