import matplotlib.pyplot as plt
import ast
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import make_interp_spline


def plot_data_languages(cleaned_data):
    # 提取和展开所有语言到一个列表中
    languages_list = cleaned_data['Movie languages'].dropna().apply(lambda x: list(ast.literal_eval(x).values()))
    flattened_languages = [lang for sublist in languages_list for lang in sublist]

    # 统计每种语言的出现次数
    language_counts = pd.Series(flattened_languages).value_counts()

    # Select the top 10 most common languages
    top_10_languages = language_counts.head(10)

    # Plot the distribution of the top 10 movie languages
    plt.figure(figsize=(10, 6))
    top_10_languages.plot(kind='bar', alpha=0.75, color='orange')
    plt.title('Top 10 Movie Languages')
    plt.xlabel('Languages')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_runtime_influence(cleaned_data):

    # 分析Movie Runtime对averageRating和Adjusted_Revenue的影响
    # 提取runtime, averageRating, Adjusted_Revenue数据
    runtime_data = cleaned_data[['Movie runtime', 'averageRating', 'Adjusted_Revenue']].dropna()

    # 绘制Runtime vs averageRating
    fig_rating = px.scatter(
        runtime_data,
        x='Movie runtime',
        y='averageRating',
        title='Movie Runtime vs Average Rating',
        labels={'Movie runtime': 'Runtime (minutes)', 'averageRating': 'Average Rating'},
        width=1000,
        height=600,
    )
    fig_rating.update_layout(
        title_font=dict(size=20),
        xaxis=dict(title='Runtime (minutes)', titlefont=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title='Average Rating', titlefont=dict(size=14), tickfont=dict(size=12))
    )
    fig_rating.show()

    # 绘制Runtime vs Adjusted_Revenue
    fig_revenue = px.scatter(
        runtime_data,
        x='Movie runtime',
        y='Adjusted_Revenue',
        title='Movie Runtime vs Adjusted Revenue',
        labels={'Movie runtime': 'Runtime (minutes)', 'Adjusted_Revenue': 'Adjusted Revenue (scaled)'},
        width=1000,
        height=600
    )
    fig_revenue.update_layout(
        title_font=dict(size=20),
        xaxis=dict(title='Runtime (minutes)', titlefont=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title='Adjusted Revenue', titlefont=dict(size=14), tickfont=dict(size=12))
    )
    fig_revenue.show()


def plot_runtime_influence_distr(cleaned_data):
    # 提取runtime, averageRating, Adjusted_Revenue数据
    runtime_data = cleaned_data[['Movie runtime', 'averageRating', 'Adjusted_Revenue']].dropna()

    # 移除runtime=1003的数据
    runtime_data = runtime_data[runtime_data['Movie runtime'] != 1003]

    # 分段：将runtime从0到380以20分钟为间隔分段
    bins = list(range(0, 381, 20))

    # 计算每个bin的averageRating和Adjusted_Revenue的平均值
    runtime_data['Runtime Bin'] = pd.cut(runtime_data['Movie runtime'], bins=bins, right=False)
    binned_data = runtime_data.groupby('Runtime Bin').agg({
        'averageRating': 'mean',
        'Adjusted_Revenue': 'mean'
    }).reset_index()

    # 移除缺失值的时间段
    binned_data = binned_data.dropna()

    # 计算每个bin的中心点，用于平滑曲线的x轴
    bin_centers = [(interval.left + interval.right) / 2 for interval in binned_data['Runtime Bin']]

    # 平滑 averageRating
    x_smooth = np.linspace(min(bin_centers), max(bin_centers), 300)
    y_avg_smooth = make_interp_spline(bin_centers, binned_data['averageRating'])(x_smooth)

    # 平滑 Adjusted_Revenue
    y_rev_smooth = make_interp_spline(bin_centers, binned_data['Adjusted_Revenue'])(x_smooth)

    # 绘制Runtime vs AverageRating的柱状图与平滑折线图
    fig_avg = go.Figure()
    # 添加柱状图
    fig_avg.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['averageRating'],
        name='Average Rating',
        marker_color='rgba(0,123,255,0.6)'
    ))
    # 添加平滑折线图
    fig_avg.add_trace(go.Scatter(
        x=x_smooth,
        y=y_avg_smooth,
        mode='lines',
        name='Smoothed Line',
        line=dict(color='green', width=2)
    ))
    fig_avg.update_layout(
        title='Runtime vs Average Rating with Smoothed Line Chart',
        xaxis_title='Runtime (minutes)',
        yaxis_title='Average Rating',
        width=1000, height=600
    )
    fig_avg.show()

    # 绘制Runtime vs Adjusted_Revenue的柱状图与平滑折线图
    fig_rev = go.Figure()
    # 添加柱状图
    fig_rev.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['Adjusted_Revenue'],
        name='Adjusted Revenue',
        marker_color='rgba(255,99,132,0.6)'
    ))
    # 添加平滑折线图
    fig_rev.add_trace(go.Scatter(
        x=x_smooth,
        y=y_rev_smooth,
        mode='lines',
        name='Smoothed Line',
        line=dict(color='green', width=2)
    ))
    fig_rev.update_layout(
        title='Runtime vs Adjusted Revenue with Smoothed Line Chart',
        xaxis_title='Runtime (minutes)',
        yaxis_title='Adjusted Revenue',
        width=1000, height=600
    )
    fig_rev.show()