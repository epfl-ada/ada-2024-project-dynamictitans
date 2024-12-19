import matplotlib.pyplot as plt
import ast
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
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

def plot_runtime_influence_distr_short(cleaned_data):
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
    # 计算每个bin的averageRating和Adjusted_Revenue的平均值和电影数
    runtime_data['Runtime Bin'] = pd.cut(runtime_data['Movie runtime'], bins=bins, right=False)
    binned_data = runtime_data.groupby('Runtime Bin').agg({
        'averageRating': 'mean',
        'Adjusted_Revenue': 'mean',
        'Movie runtime': 'count'
    }).reset_index()

    binned_data = binned_data.rename(columns={'Movie runtime': 'Movie Count'})


    # 绘制Runtime vs AverageRating与电影数量的柱状图和折线图
    fig_avg = go.Figure()
    # 添加左侧柱状图 (Average Rating)
    fig_avg.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['averageRating'],
        name='Average Rating',
        marker_color='rgba(0,123,255,0.6)',
        yaxis='y1'
    ))
    # 添加平滑折线图 (Average Rating)
    fig_avg.add_trace(go.Scatter(
        x=np.linspace(min(bin_centers), max(bin_centers), 300),
        y=y_avg_smooth,
        mode='lines',
        name='Smoothed Line (Avg Rating)',
        line=dict(color='green', width=2),
        yaxis='y1'
    ))
    # 添加右侧柱状图 (电影数)
    fig_avg.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['Movie Count'],
        name='Movie Count',
        marker_color='rgba(255,99,132,0.6)',
        yaxis='y2'
    ))
    # 更新布局，添加第二个y轴
    fig_avg.update_layout(
        title='Runtime vs Average Rating and Movie Count with Smoothed Line Chart',
        xaxis=dict(title='Runtime (minutes)'),
        yaxis=dict(title='Average Rating', side='left', showgrid=True, zeroline=True),
        yaxis2=dict(title='Movie Count', overlaying='y', side='right', showgrid=False, zeroline=True),
        width=1000, height=600,
        legend=dict(
            x=1.05,  # 进一步调整图例位置
            borderwidth=1
        )
    )
    fig_avg.show()

    # 绘制Runtime vs Adjusted_Revenue与电影数量的柱状图和折线图
    fig_rev = go.Figure()

    # 添加左侧柱状图 (Adjusted Revenue)
    fig_rev.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['Adjusted_Revenue'],
        name='Adjusted Revenue',
        marker_color='rgba(0,123,255,0.6)',
        yaxis='y1'
    ))

    # 添加平滑折线图 (Adjusted Revenue)
    fig_rev.add_trace(go.Scatter(
        x=np.linspace(min(bin_centers), max(bin_centers), 300),
        y=y_rev_smooth,
        mode='lines',
        name='Smoothed Line (Adj Revenue)',
        line=dict(color='green', width=2),
        yaxis='y1'
    ))

    # 添加右侧柱状图 (Movie Count)
    fig_rev.add_trace(go.Bar(
        x=bin_centers,
        y=binned_data['Movie Count'],
        name='Movie Count',
        marker_color='rgba(255,99,132,0.6)',
        yaxis='y2'
    ))

    # 更新布局，确保左侧和右侧 y 轴对齐且范围合理
    fig_rev.update_layout(
        title='Runtime vs Adjusted Revenue and Movie Count with Smoothed Line Chart',
        xaxis=dict(title='Runtime (minutes)', showgrid=False),
        yaxis=dict(
            title='Adjusted Revenue (Billion)',
            side='left',
            showgrid=True,
            zeroline=True,
            range=[0, binned_data['Adjusted_Revenue'].max() * 1.1]  # 使 y 轴从 0 开始
        ),
        yaxis2=dict(
            title='Movie Count',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True
        ),
        width=1000,
        height=600,
        legend=dict(
            x=1.05,  # 进一步调整图例位置
            borderwidth=1
        )
    )

    fig_rev.show()


def plot_countries_revenue_rating(cleaned_data):
    filtered_data = cleaned_data.dropna(subset=['averageRating', 'Adjusted_Revenue', 'Movie countries'])

    # 解析 'Movie countries' 并保留第一个国家
    filtered_data['Movie countries'] = filtered_data['Movie countries'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    filtered_data['Primary Country'] = filtered_data['Movie countries'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')

    # 正向计算 position_density：评分和票房的加权平均，数值越高越靠近右上角
    filtered_data['Normalized Rating'] = filtered_data['averageRating'] / filtered_data['averageRating'].max()
    filtered_data['Normalized Revenue'] = filtered_data['Adjusted_Revenue'] / filtered_data['Adjusted_Revenue'].max()
    filtered_data['Position Density'] = (filtered_data['Normalized Rating'] + filtered_data['Normalized Revenue']) / 2

    # 绘制散点图：x轴为评分，y轴为票房，颜色表示 position_density
    fig = px.scatter(
        filtered_data,
        x='averageRating',
        y='Adjusted_Revenue',
        color='Position Density',  # 正向映射
        title="Movie Ratings vs Adjusted Revenue with Position-Based Colors",
        labels={
            "averageRating": "Average Rating",
            "Adjusted_Revenue": "Adjusted Revenue (USD)",
            "Position Density": "Position Density"
        },
        hover_data=['Movie name', 'Primary Country'],  # 显示电影名称和国家
        color_continuous_scale='Reds',  # 从浅蓝到深蓝
        range_color=[0, 1],  # 映射范围设置
        width=1000,
        height=700
    )

    # 更新图表样式
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        xaxis=dict(title="Average Rating"),
        yaxis=dict(title="Adjusted Revenue"),
        coloraxis_colorbar=dict(title="Position Density")
    )

    fig.show()


def animate_country_rating_revenue(cleaned_data):
    filtered_data = cleaned_data.dropna(subset=['averageRating', 'Adjusted_Revenue', 'Movie release year', 'Movie countries'])

    # 解析 'Movie countries' 并保留第一个国家
    filtered_data['Movie countries'] = filtered_data['Movie countries'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    filtered_data['Primary Country'] = filtered_data['Movie countries'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')

    # 确保年份为整数类型，并按年份升序排序
    filtered_data['Movie release year'] = filtered_data['Movie release year'].astype(int)
    filtered_data = filtered_data.sort_values(by='Movie release year', ascending=True)

    # 获取所有年份和国家的组合，确保所有国家和年份完整覆盖
    all_years = filtered_data['Movie release year'].unique()
    all_countries = filtered_data['Primary Country'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [all_years, all_countries], names=['Movie release year', 'Primary Country']
    ).to_frame(index=False)

    # 合并原始数据，填充缺失值
    filled_data = all_combinations.merge(filtered_data, on=['Movie release year', 'Primary Country'], how='left')
    filled_data['averageRating'] = filled_data['averageRating'].fillna(0.1)  # 填充评分缺失值
    filled_data['Adjusted_Revenue'] = filled_data['Adjusted_Revenue'].fillna(0.1)  # 填充票房缺失值

    # 获取所有国家并分配唯一颜色
    unique_countries = filled_data['Primary Country'].unique()
    base_colors = pc.qualitative.Bold + pc.qualitative.D3 + pc.qualitative.Light24  # 多种高对比度调色板
    color_map = {country: base_colors[i % len(base_colors)] for i, country in enumerate(unique_countries)}

    # 强制时间轮播帧升序，保证年份逐年播放
    filled_data['Primary Country'] = filled_data['Primary Country'].astype(str)

    # 绘制时间轮播图
    fig = px.scatter(
        filled_data,
        x='averageRating',
        y='Adjusted_Revenue',
        animation_frame='Movie release year',
        color='Primary Country',
        title="Movie Ratings and Adjusted Revenue Over Time",
        labels={
            "averageRating": "Average Rating",
            "Adjusted_Revenue": "Adjusted Revenue (USD)",
            "Movie release year": "Release Year",
            "Primary Country": "Country"
        },
        hover_data=['Primary Country'],
        size='Adjusted_Revenue',
        size_max=80,
        color_discrete_map=color_map,
        template='plotly',
        width=1200,
        height=700
    )

    # 确保所有国家在图例中显示
    fig.for_each_trace(lambda t: t.update(marker=dict(opacity=0.7)))

    # 美化布局
    fig.update_layout(
        xaxis=dict(title="Average Rating", range=[0, 10]),
        yaxis=dict(title="Adjusted Revenue (USD)", range=[0, filled_data['Adjusted_Revenue'].max() * 1.1]),
        legend_title="Primary Country",
        margin=dict(l=50, r=50, t=80, b=50),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                        method="animate",
                        args=[None, {"frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True, "mode": "immediate"}]),
                    dict(
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])])
        ]
    )

    fig.show()


def plot_ratio_vs_language(cleaned_data, higher, lower):
    # List of valid languages to filter out non-language categories
    valid_languages = [
        'English', 'French', 'Spanish', 'German', 'Chinese', 'Japanese', 'Italian', 'Korean', 'Russian', 'Hindi',
        'Portuguese', 'Arabic', 'Dutch', 'Swedish', 'Turkish', 'Danish', 'Norwegian', 'Greek', 'Polish',
        'Thai', 'Finnish', 'Hebrew', 'Czech', 'Hungarian', 'Vietnamese', 'Malay', 'Indonesian', 'Romanian',
        'Bulgarian', 'Serbian'
    ]

    # Function to filter Top 30 languages
    def filter_top_languages(df, top_languages):
        return df[df['Movie languages'].apply(lambda x: any(lang in ''.join(list(ast.literal_eval(x).values())) for lang in top_languages))]

    # Clean 'Movie languages' to remove invalid entries
    language_counts = cleaned_data['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
    flattened_languages = [lang for sublist in language_counts for lang in sublist if any(valid in lang for valid in valid_languages)]

    # Extract Top 30 valid movie languages
    top_languages = pd.Series(flattened_languages).value_counts().head(30).index

    # Filter data for Top 30 languages
    higher_top30 = filter_top_languages(higher, top_languages)
    lower_top30 = filter_top_languages(lower, top_languages)
    cleaned_data_top30 = filter_top_languages(cleaned_data, top_languages)

    # Function to calculate performance ratios
    def calculate_language_performance(higher, lower, cleaned_data):
        # Count total movies per language
        language_counts = cleaned_data['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        total_count = pd.Series([lang for sublist in language_counts for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Count overperformed movies per language
        higher_languages = higher['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        overperformed_count = pd.Series([lang for sublist in higher_languages for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Count underperformed movies per language
        lower_languages = lower['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        underperformed_count = pd.Series([lang for sublist in lower_languages for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Calculate proportions
        overperformed_ratio = (overperformed_count / total_count).fillna(0)
        underperformed_ratio = (underperformed_count / total_count).fillna(0)

        return overperformed_ratio, underperformed_ratio

    # Calculate performance ratios for Top 30 languages
    over_ratio_top30, under_ratio_top30 = calculate_language_performance(higher_top30, lower_top30, cleaned_data_top30)

    # 提取前30语言的overperformed和underperformed占比
    data = pd.DataFrame({
        'Language': top_languages[:30],  # 取前30个语言
        'Overperformed': [over_ratio_top30.get(lang, 0) for lang in top_languages[:30]],
        'Underperformed': [under_ratio_top30.get(lang, 0) for lang in top_languages[:30]]
    })

    # 添加Z轴数据表示比值
    data['Ratio'] = data['Overperformed'] / data['Underperformed']

    # 使用Plotly Graph Objects模拟三维柱状图
    fig = go.Figure()

    # 添加每个语言的柱状条
    for i, row in data.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['Underperformed'], row['Underperformed']],
            y=[row['Overperformed'], row['Overperformed']],
            z=[0, row['Ratio']],
            mode='lines',
            line=dict(color=pc.qualitative.Dark24[i % len(pc.qualitative.Dark24)], width=15),
            name=row['Language']
        ))

    # 设置布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Underperformed',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    backgroundcolor='white',
                    gridcolor='black',
                    showline=True,
                    zerolinecolor='black',
                    showspikes=False),
            yaxis=dict(title='Overperformed',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    backgroundcolor='white',
                    gridcolor='black',
                    showline=True,
                    zerolinecolor='black',
                    showspikes=False),
            zaxis=dict(title='Overperformed/Underperformed',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    backgroundcolor='white',
                    gridcolor='black',
                    showline=True,
                    zerolinecolor='black',
                    showspikes=False)
        ),
        title=dict(
            text='Top 30 Movie Languages: Overperformed vs Underperformed (3D)',
            font=dict(size=20),
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            title_font=dict(size=14),
            font=dict(size=12),
            x=1.1, y=0.5
        ),
        width=1200,
        height=900
    )

    # 显示图表
    fig.show()