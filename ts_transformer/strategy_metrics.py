import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

MODEL_NAMES = ['Simple', 'ModelScaled', 'ModelScaledLongOnly', 'Model', 'MM']
METRIC_NAMES = ['Return', 'Vol', 'Sharpe', 'Downside', 'Sortino', 'MaxDrawdown', 'Calmar']

def strategy_metric(strategy, days=-1):

    df = pd.read_excel(f"single_strat_indicator/strategy_{strategy}.xlsx")
    df['Prediction_'] = df['Prediction'] - df['Previous Day Price']
    df['Actual'] = df['Expected'] - df['Previous Day Price']
    df['Simple'] = df['Actual'].cumsum()

    df['ModelScaled_'] = df['Actual'] * df['Prediction_']
    df['ModelScaled'] = df['ModelScaled_'].cumsum()

    df['ModelScaledLongOnly_'] = df['ModelScaled_']
    df.loc[df['Prediction_'] < 0, 'ModelScaledLongOnly_'] = 0
    df['ModelScaledLongOnly'] = df['ModelScaledLongOnly_'].cumsum()

    df['Model_'] = df['Actual'] * np.sign(df['Prediction_'])
    df['Model'] = df['Model_'].cumsum()

    ave_prediction = df['Prediction_'].iloc[:days].mean()
    std_prediction = df['Prediction_'].iloc[:days].std()
    df['MM_'] = (df['Prediction_'] >= (std_prediction * (-1))) * df['Actual']
    df['MM'] = df['MM_'].cumsum()

    df_weekly_cum = df[df.index % 7 == 6][MODEL_NAMES].reset_index(drop=True)
    df_weekly_pnl = df_weekly_cum.diff()
    df_weekly_pnl.iloc[0] = df_weekly_cum.iloc[0]
    df_downside_var = df_weekly_pnl.copy(deep=True)
    df_downside_var[df_downside_var>=0] = 0
    df_drawdown = df_weekly_cum.cummax()
    df_drawdown = (df_drawdown - df_weekly_cum.shift(-1).fillna(0)).shift(1)

    df_metrics = pd.DataFrame(0,index=METRIC_NAMES, columns=MODEL_NAMES)
    df_metrics.loc['Return'] = df_weekly_pnl.mean() * 52
    df_metrics.loc['Vol'] = df_weekly_pnl.std() * (52**0.5)
    df_metrics.loc['Sharpe'] = df_metrics.loc['Return']/df_metrics.loc['Vol']
    df_metrics.loc['Downside'] = df_downside_var.std() * (52**0.5)
    df_metrics.loc['Sortino'] = df_metrics.loc['Return']/df_metrics.loc['Downside']
    df_metrics.loc['MaxDrawdown'] = df_drawdown.max()
    df_metrics.loc['Calmar'] = df_metrics.loc['Return']/df_metrics.loc['MaxDrawdown']

    writer = pd.ExcelWriter(f"single_strat_indicator/strategy_{strategy}_analysis.xlsx", engine='xlsxwriter')
    df[['Prediction_', 'Actual']+MODEL_NAMES].to_excel(writer, sheet_name='Daily', index=False)
    df_weekly_cum.to_excel(writer, sheet_name='Weekly_CumPnL', index=False)
    df_weekly_pnl.to_excel(writer, sheet_name='Weekly_PnL', index=False)
    df_downside_var.to_excel(writer, sheet_name='Weekly_Downside_VAR', index=False)
    df_drawdown.to_excel(writer, sheet_name='Weekly_Drawdown', index=False)
    df_metrics.to_excel(writer, sheet_name='Weekly_Metrics')
    writer.close()

    plt.figure()
    plt.plot(df_weekly_cum, label=df_weekly_cum.columns)
    plt.title(f'Strategy_{strategy}_Weekly_CumPnL')
    plt.legend()
    plt.savefig(f'single_strat_indicator/Strategy_{strategy}_Weekly_CumPnL.png')
    #plt.show()

for i in range(1, 35):
    strategy_metric(i)