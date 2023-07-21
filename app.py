import time
import pandas as pd
import numpy as np

import dash
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import Input, Output, dcc, html

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics.pairwise import cosine_similarity

# Incorporate data
df_raw = pd.read_excel('data.xlsx')
df_raw = df_raw.iloc[:, 1:]
df_raw = df_raw.set_index('★나만 알아볼 수 있는 별명★ (이름 대신 쓸 거예요^^)')
df_raw = df_raw.astype(float)

df_nut = pd.read_excel('data_nutrition_final.xlsx')
df_nut = df_nut.iloc[:, [3,10,11,13,14,15,16,17,18,19,21,23]]
df_nut = df_nut.set_index('식품명')
df_nut = df_nut.replace('-', 0).fillna(0)
df_nut = df_nut.astype(float)
df_nut = df_nut.mul(df_nut.iloc[:, -1], axis=0).div(df_nut.iloc[:, 0], axis=0)
df_nut = df_nut.iloc[:, 1:-1]

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Options", className="display-4"),
        html.Hr(),
        html.P(
            "닉네임", className="lead"
        ),
        dcc.Dropdown(df_raw.index, id='dropdown_nickname'),
        html.Hr(),
        html.P(
            "결측치 채우기", className="lead"
        ),
        dcc.Dropdown(['평균-세로축', '평균-가로축'], id='dropdown_fillna'),
        html.Hr(),
        html.P(
            "분석할 변수들", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("홈", href="/", active="exact"),
                dbc.NavLink("날짜-영양소", href="/day-nutrition", active="exact"),
                dbc.NavLink("날짜-메뉴", href="/day-menu", active="exact"),
                dbc.NavLink("메뉴-영양소", href="/menu-nutrition", active="exact"),
                dbc.NavLink("메뉴-메뉴", href="/menu-menu", active="exact"),
                dbc.NavLink("사람-사람", href="/person-person", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    
    if pathname == "/":
        
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("데이터"),
                html.Hr(),
                html.Div([
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Bar Chart", tab_id="barchart"),
                            dbc.Tab(label="Stacked Bar Chart", tab_id="stackedbarchart"),
                            dbc.Tab(label="Line Chart", tab_id="linechart"),
                            dbc.Tab(label="Pie Chart", tab_id="piechart"),
                            dbc.Tab(label="Radar Chart", tab_id="radarchart"),
                        ],
                        id="tabs",
                        active_tab="barchart",
                    ),
                ], hidden=True),
                html.Div([
                    html.Br(),
                    html.H5("y 변수 선택"),
                    dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y', multi=True)
                ], hidden=True),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    elif pathname == "/day-nutrition":
        
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("시각화"),
                html.Hr(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Bar Chart", tab_id="barchart"),
                        dbc.Tab(label="Stacked Bar Chart", tab_id="stackedbarchart"),
                        dbc.Tab(label="Line Chart", tab_id="linechart"),
                        dbc.Tab(label="Pie Chart", tab_id="piechart"),
                        dbc.Tab(label="Radar Chart", tab_id="radarchart"),
                    ],
                    id="tabs",
                    active_tab="barchart",
                ),
                html.Div([
                    html.Br(),
                    html.H5("y 변수 선택"),
                    dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y', multi=True)
                ]),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    elif pathname == "/day-menu":
        
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("시각화"),
                html.Hr(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Box Plot", tab_id="boxplot"),
                        dbc.Tab(label="Violin Plot", tab_id="violinplot"),
                    ],
                    id="tabs",
                    active_tab="boxplot",
                ),
                html.Div([
                    html.Br(),
                    html.H5("y 변수 선택"),
                    dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y')
                ]),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    elif pathname == "/menu-nutrition":
                
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("시각화"),
                html.Hr(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Stacked Bar Chart", tab_id="stackedbarchart"),
                        dbc.Tab(label="Bubble Chart", tab_id="bubblechart"),
                    ],
                    id="tabs",
                    active_tab="stackedbarchart",
                ),
                html.Div([
                    html.Br(),
                    html.H5("y 변수 선택"),
                    dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y', multi=True)
                ]),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    elif pathname == "/menu-menu":
        
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("시각화"),
                html.Hr(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Heatmap", tab_id="heatmap"),
                    ],
                    id="tabs",
                    active_tab="heatmap",
                ),
                html.Div([dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y')], hidden=True),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    elif pathname == "/person-person":
        
        content = dbc.Container(
            [
                # dcc.Store(id="store"),
                html.H1("시각화"),
                html.Hr(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Heatmap", tab_id="heatmap"),
                    ],
                    id="tabs",
                    active_tab="heatmap",
                ),
                html.Div([dcc.Dropdown(df_nut.columns, value='에너지(Kcal)', id='dropdown_y')], hidden=True),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
        
        return content
    
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

@app.callback(Output("tab-content", "children"), [Input("dropdown_nickname", "value"), Input("url", "pathname"), Input("tabs", "active_tab"), Input('dropdown_fillna', 'value')], Input('dropdown_y', 'value'))
def generate_graphs(nickname, pathname, active_tab, fillna, y):
    
    if nickname is None:
        return '닉네임을 선택해주세요.'
    
    if fillna == '평균-세로축':
        df = df_raw.fillna(df_raw.mean(axis=0), axis=0)
    elif fillna == '평균-가로축':
        df = df_raw.T.fillna(df_raw.T.mean(axis=0), axis=0).T.round(4)
    else:
        df = df_raw

    # if not n:
    #     # generate empty graphs when app loads
    #     return {k: go.Figure(data=[]) for k in ["scatter", "hist_1", "hist_2"]}
    
    if active_tab is not None:
        
        if pathname == '/':
            content = dash_table.DataTable(data=df.to_dict('records'))
            
            return content
        
        elif pathname == '/day-nutrition':
            
            df_merge = df.stack(dropna=False).loc['test1'].reset_index()
            df_merge.columns = ['food', 'value']
            new_col = list('월' * 7 + '화' * 6 + '수' * 6 + '목' * 5 + '금' * 6)
            df_merge['day'] = new_col
            df_merge['food'] = df_merge['food'].str.replace('.1', '')
            df_merge = df_merge.merge(df_nut.drop_duplicates(), left_on='food', right_index=True)
            df_merge.iloc[:, 3:] = df_merge.iloc[:, 3:].mul(df_merge.iloc[:, 1], axis=0).div(4)
            
            df_gp = df_merge.iloc[:, 1:].groupby('day').sum()
            df_gp = df_gp.loc[list('월화수목금')]
            
            if not isinstance(y, list):
                y = [y]
            
            if active_tab == "barchart":
                
                fig = go.Figure(
                    data=[go.Bar(x=df_gp.index, y=df_gp.loc[:, y_], name=y_) for y_ in y]
                )
                
                fig.update_layout(
                    title='막대그래프(Bar Chart)',
                    xaxis=dict(
                                title='날짜',
                                # tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='Value' if len(y) > 1 else y[0]),
                    barmode='group',
                    legend=dict(
                        title='영양소',
                    )
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "stackedbarchart":
                
                fig = go.Figure(
                    data=[go.Bar(x=df_gp.index, y=df_gp.loc[:, y_], name=y_) for y_ in y]
                )
                
                fig.update_layout(
                    title='누적막대그래프(Stacked Bar Chart)',
                    xaxis=dict(
                                title='날짜',
                                # tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='Value' if len(y) > 1 else y[0]),
                    barmode='stack',
                    legend=dict(
                        title='영양소',
                    )
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "linechart":
                
                fig = go.Figure(
                    data=[go.Line(x=df_gp.index, y=df_gp.loc[:, y_], name=y_) for y_ in y]
                )
                
                fig.update_layout(
                    title='선그래프(Line Chart)',
                    xaxis=dict(
                                title='날짜',
                                # tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='Value' if len(y) > 1 else y[0]),
                    barmode='group',
                    legend=dict(
                        title='영양소',
                    )
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "piechart":

                # subplots를 이용하여 1x3 구조 생성
                fig = make_subplots(rows=1, cols=5, subplot_titles=list('월화수목금'),
                                    specs=[[{'type':'domain'}] * 5])
                                
                for i, day in enumerate(list('월화수목금')):
                    fig.add_trace(go.Pie(labels=y, values=df_gp.loc[day, y], name=day, sort=False), row=1, col=i + 1)                    
                                    
                fig.update_layout(
                    title='파이차트(Pie Chart)',
                )
                    
                return dcc.Graph(figure=fig)
                
            elif active_tab == "radarchart":

                fig = go.Figure()

                for i, day in enumerate(list('월화수목금')):
                    fig.add_trace(go.Scatterpolar(
                        r=df_gp.loc[day, y],
                        theta=y,
                        fill='toself',
                        name=day
                    ))

                fig.update_layout(
                    title='레이더차트(Radar Chart)',
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 50])
                    ),
                    showlegend=True
                )
                
                return dcc.Graph(figure=fig)
            
        elif pathname == '/day-menu':
            
            df_merge = df.stack(dropna=False).reset_index()
            df_merge.columns = ['nickname', 'food', 'value']
            new_col = list('월' * 7 + '화' * 6 + '수' * 6 + '목' * 5 + '금' * 6)
            df_merge['day'] = new_col * int(len(df_merge.index) / len(new_col))
            df_merge['food'] = df_merge['food'].str.replace('.1', '')
            df_merge = df_merge.merge(df_nut.drop_duplicates(), left_on='food', right_index=True)
            df_merge.iloc[:, 4:] = df_merge.iloc[:, 4:].mul(df_merge.iloc[:, 2], axis=0).div(4)
            
            df_gp = df_merge.iloc[:, :].groupby(['day', 'nickname']).sum()
            df_gp = df_gp.drop('food', axis=1)
            
            if active_tab == "boxplot":

                fig = go.Figure()

                for i, day in enumerate(list('월화수목금')):
                    fig.add_trace(go.Box(
                        y=df_gp.loc[day, y],
                        name=day,
                    ))
                    fig.add_trace(
                        go.Scatter(x=[day], y=[df_gp.loc[(day, nickname), y]], mode='markers', marker=dict(color='red', size=10))
                    )

                fig.update_layout(
                    title='박스플롯(Box Plot)',
                    xaxis=dict(
                                title='날짜',
                                # tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title=y),
                    showlegend=False,
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "violinplot":

                fig = go.Figure()

                for i, day in enumerate(list('월화수목금')):
                    fig.add_trace(go.Violin(
                        y=df_gp.loc[day, y],
                        name=day,
                    ))
                    fig.add_trace(
                        go.Scatter(x=[day], y=[df_gp.loc[(day, nickname), y]], mode='markers', marker=dict(color='red', size=10))
                    )

                fig.update_layout(
                    title='바이올린플롯(Violin Plot)',
                    xaxis=dict(
                                title='날짜',
                                # tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title=y),
                    showlegend=False,
                )
                
                return dcc.Graph(figure=fig)
                    
        elif pathname == '/menu-nutrition':
            
            if not isinstance(y, list):
                y = [y]
            
            if active_tab == "stackedbarchart":
                
                fig = go.Figure(
                    data=[go.Bar(x=df_nut.drop_duplicates().index, y=df_nut.drop_duplicates().loc[:, y_], name=y_) for y_ in y]
                )
                
                fig.update_layout(
                    title='누적 막대그래프(Stacked Bar Chart)',
                    xaxis=dict(
                                title='날짜',
                                tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='Value' if len(y) > 1 else y[0]),
                    barmode='stack',
                    legend=dict(
                        title='영양소',
                    )
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "bubblechart":

                fig = go.Figure()

                # df_stack = df_nut.div(df_nut.max(), axis=1) * 10
                df_stack = np.log10(df_nut + 1)
                # df_stack = df_nut
                df_stack = df_stack.stack()
                df_stack = df_stack.reset_index()
                df_stack.columns = ['food', 'nutrition', 'value']

                # 버블 차트 그리기
                fig = go.Figure()

                for cat1 in df_stack['nutrition'].unique():
                    for cat2 in df_stack['food'].unique():
                        filtered_data = df_stack[(df_stack['nutrition'] == cat1) & (df_stack['food'] == cat2)]
                        fig.add_trace(go.Scatter(
                            x=[cat1],
                            y=[cat2],
                            mode='markers',
                            marker=dict(size=filtered_data['value'] * 20)  # 버블 크기를 Value의 5배로 설정
                        ))

                fig.update_layout(
                    title='버블차트(Bubble Chart)',
                    xaxis=dict(
                                title='영양소',
                                tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(
                        title='메뉴',
                        dtick=1,
                    ),
                    height=1000,
                    showlegend=False,
                )
                
                return dcc.Graph(figure=fig)
                        
        elif pathname == '/menu-menu':
            
            df_rescale = df_nut.drop_duplicates().div(df_nut.max(), axis=1) * 10
            
            if active_tab == "heatmap":
                fig = go.Figure(
                    data=[go.Heatmap(z=cosine_similarity(df_rescale), x=df_rescale.index, y=df_rescale.index, colorscale='Tealrose')]
                )

                fig.update_layout(
                    title='히트맵(Heatmap)',
                    xaxis=dict(
                                title='닉네임',
                                tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='닉네임'),
                    height=700,
                )
                return dcc.Graph(figure=fig)
                
        elif pathname == '/person-person':
            
            if active_tab == "heatmap":
                fig = go.Figure(
                    data=[go.Heatmap(z=cosine_similarity(df), x=df.index, y=df.index, colorscale='Tealrose')]
                )

                fig.update_layout(
                    title='히트맵(Heatmap)',
                    xaxis=dict(
                                title='닉네임',
                                tickangle=-90,
                                tickfont=dict(size=10)
                            ),
                    yaxis=dict(title='닉네임'),
                    height=700,
                )
                return dcc.Graph(figure=fig)
            
    return "No tab selected"

if __name__ == "__main__":
    app.run_server(port=8888, debug=False)