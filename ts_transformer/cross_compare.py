import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import seaborn as sns

MODEL_NAMES = ['Simple', 'ModelScaled', 'ModelScaledLongOnly', 'Model', 'MM']
METRIC_NAMES = ['Return', 'Vol', 'Sharpe', 'Downside', 'Sortino', 'MaxDrawdown', 'Calmar']

def cross_compare(strategies, metric_name):
    df = pd.DataFrame(columns=MODEL_NAMES)
    for i in strategies:
        df_read = pd.read_excel(f"single_strat_indicator/strategy_{i}_analysis.xlsx", sheet_name='Weekly_Metrics', index_col=0)
        df.loc[f"strategy_{i}"] = df_read.loc[metric_name]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='YlGnBu', annot=True, fmt=".2f", cbar=True)

    plt.xlabel('Model_Construct_Method')
    plt.ylabel('Strategy_No.')
    plt.title(f'Heat Map for {metric_name}')
    plt.savefig(f'single_strat_indicator/All_strategies_{metric_name}.png')
    #plt.show()

for i in METRIC_NAMES:
    cross_compare(range(1,35), i)