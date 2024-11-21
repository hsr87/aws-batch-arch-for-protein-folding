import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np

def create_correlation_plot(df_prev: pd.DataFrame, df_curr: pd.DataFrame):
    """예측 결과 시각화를 위한 Plotly 그래프 생성"""
    
    # 데이터 병합
    merged_df = pd.merge(df_prev, df_curr, 
                        on='input', 
                        suffixes=('_previous', '_current'))
    
    # 통계 계산
    correlation = np.corrcoef(merged_df['output_previous'], 
                            merged_df['output_current'])[0,1]
    r_squared = correlation**2
    rmse = np.sqrt(mean_squared_error(merged_df['output_previous'], 
                                    merged_df['output_current']))
    
    # Plotly 그래프 생성
    fig = go.Figure()
    
    # 산점도 추가
    fig.add_trace(go.Scatter(
        x=merged_df['output_previous'],
        y=merged_df['output_current'],
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))
    
    # y=x 라인 추가
    min_val = min(merged_df['output_previous'].min(), 
                 merged_df['output_current'].min())
    max_val = max(merged_df['output_previous'].max(), 
                 merged_df['output_current'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='y=x',
        line=dict(
            color='red',
            dash='dash'
        )
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title='Correlation between Previous and Current Predictions',
        xaxis_title='Previous Predictions',
        yaxis_title='Current Predictions',
        showlegend=True,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'Statistics:<br>R² = {r_squared:.3f}<br>RMSE = {rmse:.3f}<br>Pearson r = {correlation:.3f}',
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        ]
    )
    
    return fig, r_squared, rmse, correlation