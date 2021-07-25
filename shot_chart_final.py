import pandas as pd
import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

test_png = '1.png'
png2 = '2.png'
png3 = '3.png'
png4 = '4.png'
png5 = '5.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')
png2b = base64.b64encode(open(png2, 'rb').read()).decode('ascii')
png3b = base64.b64encode(open(png3, 'rb').read()).decode('ascii')
png4b = base64.b64encode(open(png4, 'rb').read()).decode('ascii')
png5b = base64.b64encode(open(png5, 'rb').read()).decode('ascii')

# Wczytanie danych
df = pd.read_csv(r'C:\Users\kiszk\Desktop\python\dane\nba_shot_data_2000-2020.csv')
df_2 = pd.read_csv(r'C:\Users\kiszk\Desktop\python\dane\top.csv', delimiter=';')
df_3 = pd.read_csv('gg.csv')
df_3['Season'] = '2019/20'
df_mod = df_3.drop_duplicates(subset=['Player'])
df_4 = pd.read_csv('gg1.csv')
df_4['Season'] = '2000/01'
df_1_mod = df_4.drop_duplicates(subset=['Player'])
frames = [df_mod, df_1_mod]
df_res = pd.concat(frames)


# Liczenie zakresów do piramidy
def zakresy(nazwa, a, b):
    df_res[nazwa] = np.where((df_res['3P'] * df_res['G'] > a) &
                             (df_res['3P'] * df_res['G'] <= b), 1, 0)


df_res['0-15'] = np.where(df_res['3P'] * df_res['G'] <= 15, 1, 0)
for i in range(15, 200, 15):
    zakres_dolny = i + 1
    zakres_gorny = i + 15
    zakres = str(zakres_dolny) + '-' + str(zakres_gorny)
    zakresy(zakres, i, i + 15)
df_res['wiecej_niz_210'] = np.where(df_res['3P'] * df_res['G'] > 210, 1, 0)

df_2020 = df_res[df_res['Season'] == '2019/20']
df_2000 = df_res[df_res['Season'] == '2000/01']
list_2020 = []
for i in range(0, 15):
    total = df_2020.iloc[:, i - 15].sum()
    list_2020.append(total)
list_2000 = []
for i in range(0, 15):
    total = df_2000.iloc[:, i - 15].sum()
    list_2000.append(total)
np_list_2020 = np.array(list_2020)
np_list_2000 = np.array(list_2000)
np_list_2000 = np_list_2000 * (-1)
tick_list = []
tick_text_list = []
for i in range(0, 210, 15):
    x = str(i) + '-' + str(i + 15)
    tick_text_list.append(x)
for i in range(7, 210, 15):
    tick_list.append(i)
np_tick_list = np.array(tick_list)
tick_text_list[-1] = '210+'
y = list(range(7, 240, 15))
# Przygotowanie wykresu piramidy
layout = go.Layout(yaxis=go.layout.YAxis(range=[0, 220],
                                         tickvals=np_tick_list,
                                         ticktext=tick_text_list,
                                         title='Number of 3-pointers made'),
                   xaxis=go.layout.XAxis(
                       range=[-350, 350],
                       tickvals=[-300, -200, -100, 0, 100, 200, 300],
                       ticktext=[300, 200, 100, 0, 100, 200, 300],
                       title='Number of players'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=y,
               x=np_list_2020,
               orientation='h',
               name='2019/20 Season',
               hoverinfo='x',
               marker=dict(color='PowderBlue')
               ),
        go.Bar(y=y,
               x=np_list_2000,
               orientation='h',
               name='2000/01 Season',
               text=-1 * np_list_2000.astype('int'),
               hoverinfo='text',
               marker=dict(color='red')
               )]

fig_pyramid = go.Figure(data, layout)

# Przygotowanie dashboard'a
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Linear chart', children=[
            html.Img(src='data:image/png;base64,{}'.format(test_base64),
                     className='align-self-center',
                     style={
                         'height': '50%',
                         'width': '50%'
                     })
        ]),
        dcc.Tab(label='Histogram 2PT made', children=[
            html.Img(src='data:image/png;base64,{}'.format(png2b),
                     className='align-self-center',
                     style={
                         'height': '50%'
                     })
        ]),
        dcc.Tab(label='Histogram 3PT made', children=[
            html.Img(src='data:image/png;base64,{}'.format(png3b),
                     className='align-self-center',
                     style={
                         'height': '50%'
                     })
        ]),
        dcc.Tab(label='Pie chart', children=[
            html.Img(src='data:image/png;base64,{}'.format(png4b),
                     className='align-self-center',
                     style={
                         'height': '50%'
                     })
        ]),
        dcc.Tab(label='Bar chart', children=[
                    html.Img(src='data:image/png;base64,{}'.format(png5b),
                             className='align-self-center',
                             style={
                                 'height': '50%'
                             })
                ]),

        dcc.Tab(label='Pyramid chart', children=[
            html.H1("3-PT Shots made distribution by all players in 2000/01 and 2019/20 seasons",
                    style={'text-align': 'center'}),
            dcc.Graph(
                figure=
                fig_pyramid,
                style={'width': '150vh', 'height': '80vh', 'display': 'inline-block', "position": "absolute",
                       "top": "15%",
                       "left": "13%"}

            )
        ]),
        dcc.Tab(label='Sunburst chart', children=[
            html.H1("Top 15 3-pt made in single season by player", style={'text-align': 'center'}),
            dcc.Graph(
                figure=
                px.sunburst(df_2, path=['Season', 'Player', '3P'], values='3pt'),
                style={'width': '150vh', 'height': '80vh', 'display': 'inline-block', "position": "absolute",
                       "top": "18%",
                       "left": "12%"}

            )
        ]),
        dcc.Tab(label='TvsT shot chart', children=[
            html.H1("Team vs Team shot chart", style={'text-align': 'center'}),
            html.Label("Season:"),
            dcc.Dropdown(id="slct_year",
                         options=[
                             {'label': i, 'value': i} for i in df['season'].unique()],
                         multi=False,
                         value='2019/2020',
                         style={'width': "40%"}
                         ),
            html.Label("Team:"),
            dcc.Dropdown(id='teams_dropdown', options=[], style={'width': "40%"}),
            html.Label("Opponent team:"),
            dcc.Dropdown(id='enemy_teams_dropdown', options=[], style={'width': "40%"}),
            html.Label("Date:"),
            dcc.Dropdown(id='date', options=[], style={'width': "40%"}),
            dcc.Graph(id='shot_chart', figure={}, style={'width': '200vh', 'height': '90vh'})
        ]),
        dcc.Tab(label='Off vs Def chart', children=[
            html.H1("Offensive FG% vs Defensive FG% zone chart", style={'text-align': 'center'}),
            html.Label("Season:"),
            dcc.Dropdown(id="select_season",
                         options=[
                             {'label': i, 'value': i} for i in df['season'].unique()],
                         multi=False,
                         value='2019/2020',
                         style={'width': "40%"}
                         ),
            html.Label("Team:"),
            dcc.Dropdown(id='teams_dropdown_zone', options=[], style={'width': "40%"}),
            dcc.Graph(id='shot_zone', figure={}, style={'width': '200vh', 'height': '90vh'})
        ]),
        dcc.Tab(label='Points per zone chart', children=[
            html.H1("Points per zone chart by season", style={'text-align': 'center'}),
            dcc.Dropdown(id="select_season_per_zone",
                         options=[
                             {'label': i, 'value': i} for i in df['season'].unique()],
                         multi=False,
                         value='2019/2020',
                         style={'width': "40%"}
                         ),
            dcc.Graph(id='shot_zone_season',
                      figure={}, style={'width': '200vh', 'height': '90vh'}
                      )
        ]),
    ])
])


# Łączenie wykresów interaktywnych z dashboard'em


@app.callback(
    Output(component_id='teams_dropdown', component_property='options'),
    Input(component_id='slct_year', component_property='value')
)
def set_teams_options(chosen_season):
    dff = df[df.season == chosen_season]
    return [{'label': t, 'value': t} for t in sorted(dff.team.unique())]


@app.callback(
    Output(component_id='teams_dropdown', component_property='value'),
    Input(component_id='teams_dropdown', component_property='options')
)
def set_teams_value(available_options):
    first_dict = available_options[0]
    return first_dict['value']


@app.callback(
    Output(component_id='enemy_teams_dropdown', component_property='options'),
    Input(component_id='teams_dropdown', component_property='value'),
    Input(component_id='slct_year', component_property='value')
)
def set_enemy_teams_options(chosen_team, chosen_season):
    dff = df[df.season == chosen_season]
    dff = dff[dff.team == chosen_team]
    return [{'label': e, 'value': e} for e in sorted(dff.enemy_team.unique())]


@app.callback(
    Output(component_id='enemy_teams_dropdown', component_property='value'),
    Input(component_id='enemy_teams_dropdown', component_property='options')
)
def set_teams_value(available_options_1):
    first_dict = available_options_1[0]
    return first_dict['value']


@app.callback(
    Output(component_id='date', component_property='options'),
    Input(component_id='enemy_teams_dropdown', component_property='value'),
    Input(component_id='teams_dropdown', component_property='value'),
    Input(component_id='slct_year', component_property='value')
)
def set_enemy_teams_options(enemy_chosen_team, chosen_team, chosen_season):
    dff = df[df.season == chosen_season]
    dff = dff[dff.team == chosen_team]
    dff = dff[dff.enemy_team == enemy_chosen_team]
    return [{'label': e, 'value': e} for e in sorted(dff.date.unique())]


@app.callback(
    Output(component_id='date', component_property='value'),
    Input(component_id='date', component_property='options')
)
def set_teams_value(available_options_2):
    first_dict = available_options_2[0]
    return first_dict['value']


@app.callback(
    Output(component_id='shot_chart', component_property='figure'),
    Input(component_id='slct_year', component_property='value'),
    Input(component_id='teams_dropdown', component_property='value'),
    Input(component_id='enemy_teams_dropdown', component_property='value'),
    Input(component_id='date', component_property='value')
)
# Przygotowanie danych do wykresu rzutów oraz przygotowanie wykresu
def update_graph(choosen_season, choosen_team, chosen_enemy_team, date):
    dff = df.copy()
    dff_enemy = df.copy()
    dff = dff[dff["season"] == choosen_season]
    dff = dff[dff["team"] == choosen_team]
    dff = dff[dff['enemy_team'] == chosen_enemy_team]
    dff = dff[dff['date'] == date]
    df_missed = dff[dff['outcome'] == 'missed']
    df_made = dff[dff['outcome'] == 'made']
    dff_enemy = dff_enemy[dff_enemy["season"] == choosen_season]
    dff_enemy = dff_enemy[dff_enemy["team"] == chosen_enemy_team]
    dff_enemy = dff_enemy[dff_enemy["enemy_team"] == choosen_team]
    dff_enemy = dff_enemy[dff_enemy["date"] == date]
    dff_enemy['x'] = 1000 - dff_enemy['x']
    dff_enemy['y'] = -dff_enemy['y']
    dff_enemy_missed = dff_enemy[dff_enemy["outcome"] == 'missed']
    dff_enemy_made = dff_enemy[dff_enemy["outcome"] == 'made']

    fig = go.Figure(go.Scatter(y=df_missed['y'], x=df_missed['x'],
                               mode='markers', marker=dict(symbol='x-thin-open', color='red', size=10), name='Shots '
                                                                                                             'missed'))
    fig.add_traces(go.Scatter(y=df_made['y'], x=df_made['x'],
                              mode='markers', marker=dict(symbol='circle-open', color='green', size=10), name='Shots '
                                                                                                              'made'))
    fig.add_traces(go.Scatter(y=dff_enemy_missed['y'], x=dff_enemy_missed['x'],
                              mode='markers', marker=dict(symbol='x-thin-open', color='orange', size=10),
                              name='Opponent team shots '
                                   'missed'))
    fig.add_traces(go.Scatter(y=dff_enemy_made['y'], x=dff_enemy_made['x'],
                              mode='markers', marker=dict(symbol='circle-open', color='blue', size=10),
                              name='Opponent team shots '
                                   'made'))

    # Rysowanie boiska
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    def draw_rect(x0, y0, x1, y1):
        fig.add_shape(type="rect",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="DarkGreen", width=3))

    fig.add_layout_image(
        dict(
            source="https://az1.hatstoremedia.com/hatstore/images/image-atlanta-hawks-kepsar-2017-02-21-125307966/555/555/0/atlanta-hawks-kepsar.png",
            xref="x",
            yref="y",
            x=452.5,
            y=47.5,
            sizex=95,
            sizey=95,
            sizing="stretch",
            opacity=1,
            layer="below")
    )
    fig.add_annotation(x=400, y=200, font=dict(size=20, color='DarkGreen'),
                       text=choosen_team,
                       showarrow=False,
                       yshift=10)
    fig.add_annotation(x=600, y=200, font=dict(size=20, color='DarkGreen'),
                       text=chosen_enemy_team,
                       showarrow=False,
                       yshift=10)

    draw_rect(-45, -255, 1045, 255)
    draw_rect(-45, -80, 140, 80)
    draw_rect(860, -80, 1045, 80)
    fig.add_shape(type="line", x0=500, y0=-255, x1=500, y1=-42, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=500, y0=42, x1=500, y1=255, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=-220, x1=91, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=220, x1=91, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=-220, x1=909, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=220, x1=909, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(y_center=0, a=240, b=-238, start_angle=2.26 * np.pi / 6,
                                   end_angle=-2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(x_center=1000, a=-240, b=238, start_angle=-2.26 * np.pi / 6,
                                   end_angle=2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    return fig


@app.callback(
    Output(component_id='teams_dropdown_zone', component_property='options'),
    Input(component_id='select_season', component_property='value')
)
def set_teams_zone_options(chosen_season):
    dff = df[df.season == chosen_season]
    return [{'label': t, 'value': t} for t in sorted(dff.team.unique())]


@app.callback(
    Output(component_id='teams_dropdown_zone', component_property='value'),
    Input(component_id='teams_dropdown_zone', component_property='options')
)
def set_teams_zone_value(available_options):
    first_dict = available_options[0]
    return first_dict['value']


@app.callback(
    Output(component_id='shot_zone', component_property='figure'),
    Input(component_id='select_season', component_property='value'),
    Input(component_id='teams_dropdown_zone', component_property='value'),

)
# Przygotowanie danych i wykresu do wykresu skutecznści rzutów i obrony
def update_graph(chosen_season, chosen_team):
    dff = df.copy()
    dff_def = df.copy()
    dff = dff[dff["season"] == chosen_season]
    dff = dff[dff["team"] == chosen_team]
    dff_def = dff_def[dff_def["season"] == chosen_season]
    dff_def = dff_def[dff_def["enemy_team"] == chosen_team]

    dff['procentage'] = 0
    dff_def['procentage'] = 0

    def fg_procentage(df_def, column):
        dff_count = df_def[df_def[column] == 1]
        count_miss = dff_count[dff_count["outcome"] == "missed"].count()[0]
        count_made = dff_count[dff_count["outcome"] == "made"].count()[0]
        proctenage = count_made / (count_made + count_miss)
        df_def['procentage'] = np.where(df_def[column] == 1, proctenage, df_def['procentage'])

    fg_procentage(dff, 'left_corner')
    fg_procentage(dff, 'right_corner')
    fg_procentage(dff, 'middle_left_3pt')
    fg_procentage(dff, 'middle_right_3pt')
    fg_procentage(dff, 'middle_3pt')
    fg_procentage(dff, 'deep_2pt_left_corner')
    fg_procentage(dff, 'deep_2pt_right_corner')
    fg_procentage(dff, 'deep_2pt_left_middle')
    fg_procentage(dff, 'deep_2pt_right_middle')
    fg_procentage(dff, 'deep_2pt_middle')
    fg_procentage(dff, 'close_paint')

    fg_procentage(dff_def, 'left_corner')
    fg_procentage(dff_def, 'right_corner')
    fg_procentage(dff_def, 'middle_left_3pt')
    fg_procentage(dff_def, 'middle_right_3pt')
    fg_procentage(dff_def, 'middle_3pt')
    fg_procentage(dff_def, 'deep_2pt_left_corner')
    fg_procentage(dff_def, 'deep_2pt_right_corner')
    fg_procentage(dff_def, 'deep_2pt_left_middle')
    fg_procentage(dff_def, 'deep_2pt_right_middle')
    fg_procentage(dff_def, 'deep_2pt_middle')
    fg_procentage(dff_def, 'close_paint')
    dff_def['x'] = 1000 - dff_def['x']
    dff_def['y'] = -dff_def['y']

    # dff['procentage'] = np.log(dff['procentage'])
    # dff_def['procentage'] = np.log(dff_def['procentage'])
    fig = go.Figure(data=go.Scatter(
        x=dff['x'],
        y=dff['y'],
        showlegend=False,
        mode='markers',
        marker=dict(
            symbol='hexagon2',
            size=12,
            color=dff['procentage'],
            colorscale='rdylbu',
            showscale=True,
            reversescale=True
        ),
        marker_colorbar_showticklabels=False
    ))
    fig.add_traces(go.Scatter(y=dff_def['y'],
                              x=dff_def['x'], showlegend=False,
                              mode='markers', marker=dict(symbol='hexagon2', color=dff_def['procentage'],
                                                          colorscale='rdylbu', showscale=True,
                                                          reversescale=True, size=12),
                              marker_colorbar_showticklabels=True,
                              marker_colorbar_title=dict(text='FG%'),
                              marker_colorbar_tickvals=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                              marker_colorbar_ticktext=['20%', '25%', '30%', '35%', '40%', '45%', '50%', '55%', '60%'],
                              marker_colorbar_title_font_size=11,
                              marker_colorbar_title_font_color='white'
                              ))

    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    def draw_rect(x0, y0, x1, y1):
        fig.add_shape(type="rect",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="DarkGreen", width=3))

    fig.add_layout_image(
        dict(
            source="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAN0AAADkCAMAAAArb9FNAAAAk1BMVEX8/v8tWDn///8pVjYkUzIgUS8ZTioVTCceUC0nVTURSiQjUjEJSB/4+vsTSyXT2dQARhzs7+3J0MmzvbLi5uKfrJ/z9fTL0su+x740Wzrq7uwARh1ogGpAY0V/koClsaXa39pYc1pvhXGNno56jnust6w9YUNJaUyYpphTcFcmUioAOwBgemLCysEWSyAARBU0WjUbq/9KAAAaQUlEQVR4nO1d6baqOgzWlNmCKIriccb5qHu//9PdpmUWFQp7uGud/DiDStuPtEmaJmmn84/+0T/6R//otxHk6aeH0w5FWGx7tp4MI5r01wO78/+Gycfuj277cHsN7i61HEa6rrM/HIvSy+G6Db3hzP7/QcQRDybj3cGguqYqJuk+EjEVQ3Oosliu+v7/BiIOdOTtAlc3lAIqIqjwYc9w3MVyOvv9AHEyTncB1XokYZHKWGRR171fgsPxuDh8Xu7uHzZRNaOXMJWoOl0sb/YvRohMW52pHiMzVc3SD5tw7PXXIzsnMv3RerIfL8+BY2mqmSK8eoPfCRBgNj5Sw4ymm0Yv13C6HjxRAfHHg7633NwtLZrEpkave/+34WPj8TZWBE3RrGDpjSpJfPGb9X6nMIQRQGd++00MBFiHui6gqZay9WY1ZSCXRPu54/TE63E+T7Nfgg/gdqWqmI/WJexLigZcjbflXeccJIa17f8CfABeYJnijXfDfrMpxZ4e7gwxCxR6vP0wPuisAp3wl+3Oh50WRgNgTzcunwqmtZj+ID6AvaEhNlMPxoPWBsLWYCjaJfrhp/ABTAXfTHqetivkGAP3Bz7fiXWc/AA+gP7RInx9XL9i/TNZdeT4TDoffTc8GGzdqO/1F71bgMmZ41M+Qvs78QGsNJVju34VNtHN8Ojg/DDINy4/GB11XBPOl68J1DcaX37zwffAAzjxSamq3je8UOic7grXpt/RGzIO36bpLv1vepuzOcXpaV2/vEOm4jjjtOP6+xYC3AKVW+e3L14H9txCxn2Mv1UJgb10GfsIDb+yW1gTlTPu+zXQJBA9z76sZ/BwARD3S9/gs77tLeWzc/JV2nWJs1Ix+j9k+XnUxNm5+oruwd6grNTlJVdTTzSMFjg7rV37UwdmB9w4y85K3Jv2vfEpPJ1Wt5EkQOhscfZom7YNM1jjnpI4cjIZXROBa2mGoRqGoVO69eRcQzDGxdcL2jVcYHhn8kQhUrKSmVNHauSctIquhVIjFAMx5QbyrM2pi6bXUWrJQX/hmI+edpVK6UwYBQo6ztozJsBDcNpccjK5JdiQ9LPM2wL/yAQAubcluZkoxrHs5JZcSMux4fo5SMGzNwaKgHYUH3gfKIdDOXDj5+DY8jt2ZODBHHXTRxvcgymOj57kwE3cF+DY4rvKNbvVkHvN1x7ccHxWJXB2hoT3z7+UHd1lSFviz5j+yz37flQ7hKc3lZzQr845uH7QmNwLPwwYvmYdI3cKzFIIFzSlvxUsZcE9pZnegxHqKafamoOxkV1TdMM09lh/DY4cGJdXVM2wmASVXuWcdWZKiaWkCZ8waa4vqzUBfSs38h5h8M5P1EFEtA9Fqdqr1h1cmdHZOzdBd2aa05hXbcFX80PvXQEmecQFUpgOvRVmrzathq6zwMHJ6SnewNao9XrgqhQ4MwQ4vJIrOlucn4UfuBUXE/gBmxfWWHbHsWLv3QyqT23YawXWbNmmsMDQPHz7gblm5bcJIwdfxlBOqfRxytxrSF0YFEXkBWBldJ8SOUBeFDEyqu9OhTZ1ZJwR4KMgo7XsHbgWZIgLMC3wM8enDZMpBd5aNQYrZtexPrgObNgicurNapg6+aF+sN3PK3RngFMenVJZhvEOd0ZsEdQiODFN1avVFT6VlyHk+DjzcnR5QF9vsnRgYQqLoN4w0UYhQQWjKP+Yl5MRxgkeJmsey6iwVqvLlKhDLlmMujZLgIuuvpnKNWRCzgAGLyYmGxbbwe6yU5PWNR1hyt6nUs8Yh6VRe9GJB2dOOjf/7NmyejUxmaUIwh6KwdV35/GlZ+1rPAcTNi+VjdRWfB0FnXRNl83LmfMEV8w89puREnEP/cD1e7S7pJ6k5fNSkzDAeXDUlmpqT6MLPNw7vjYzhTnj7+662jPo4gYSYRNcRNSYmxCyxeLUYXaMbHRikGDmhcsVhq7Ym94bcOytD/GNTOMnTv3a7lwIcW5WlZsw+qg9L8UIA8tgW8GEJl3lLTjGvTCN/+sHmqXtvLqxmjjV9IryHTZondYRXsi0I9X5Lk0LxjwAbjDd0Dcb84gMclqL13PlTyiauwjrnMdzMWFU3DihjNXq+FFgMI9DF7s8xoscFoGjv1tyCbEnFP6Ekn5CN3UM3G2vqv4Cxmdm3tYBdykur4f44PcIi08oH9XhwcCpuJZgxUQKrXNeAOEb94IUEaPGyoexxsXv2x/697q6H/bHi6X1Kk/E92SqGu1u6mj2ihOO70hqmmBMY6295flu6WoxjL0uEcVwqHMNp6N6mo8LC9178wifwWp9ZwWX5+vbaXdQLV0zlNqMNBVV0y31sBsPRzInmOieeutN45qxqmejFKI9muxP8/MnpbquqU9SLVJWqQyUbtHLeR7uJ1Hku1TfqBXeMA98th9R6+8Gc20IkAzlbR8ud+fPC6XU4uTwbBkH/0ktSu+fx+0yXN0mI99unBVUgXl81d3bOdnMpB/Yg9FovV73JxOe5NRn/x7N/MwP2ugOzU3nJfNso7K3tF7XJdR6H8zEMo8vmuW6rpYNVtKGVO5ZC4BhyMSm9cJxgVpD2TYDt7+ctydvOBq8Z1LKx8Fs4p22p2b4YEFeOZ3gZkm5GzItdOZWlHqmdxeb5WnsTbLt5Ybvj/r96eoUXo+BxjPZ5I6a08Y9ZjO5T2ceenjqum3yDcwOqZOEEFNVDc26pF/b61GKDxZ/LEc3VNVMTEzFaHTgCExqqM+29zBDlVHXe5ZtYF22L7j48dfjC3WDZGE8HCDgG6liKz7vH32jxtMvDfbCGzTeL9vQkSD+eoW+PZIcd5cffn00ebsj6zl7UKY8ZWyFttf3MrPEjC1ybL6b2ag8OT+ptT0pDuGqPNsB8MPF2v7E9PGZXmpzxQo2cY99xOj65eEQrnwsA0yZXKGl7jFY9tA1LtuyfSi1nJVFDCZG5yYLb17KPKLL20rAtm9a6c6Jf1PXEZY+fS11fylq/CZhEO1x7wk6+1AK76XB8WYUjEOlj8MQuSr72mBV5pYl1ibRYInP/Z7qBHtLyxhe0QFUNgxcXW7J1MRFbkq5nzuRtHrgm5XLwPIjn/slq92HR1riFXTlo6SY6CqdgN0nn1ehR+mu6Po8nx3oRzA+sx8C9HeaUwSYaJH6AynnEeep1CFt5+Fgq9vT9e20GF9qk2jk+Y/Z/ui2U6z8CqzlcMy1NnT40VPxY1TlC9kJEWSjaTS6w7z4h467/EclnbDfTpK0YEFU1uLEPdyjQse5pUq+sVysg77wyh0+kTYvN2TZPuF2zkwAaasC5krJDhVdDpakHoUM6+jq2TYmOnU2n3kTAbzM2Z/jl//q7VjYm35wQMCN6YO7ZIOZWAD63EyM0L3aga1T7pXr5AqDGdFHiwsdKvXCDTLPpkfj+ovj2hjdC3eiiAXtPlmeFUfDth5awb0CR1InDib3qJ+c67/0YePW+Q26rGUtu4uGnfLg1fM16WWXicl46QjFF/gOHVMb8dKTFnHew8Lj2k6TW8iwjZXxa0EXoXvtckvFr6xBD2vcg+deMzYqa7ymETi5xcz9RRgHHPuNInsmQpf5Nidk7WSau3WDZeIWMEYqZ8rxySoZtT6wSt/2yFtuPu+uez9sPZ63E6ETawIG+/kndek9uIbTWba5eTwTpPUT6ycvQvATSSMTJrE+yPoKYfBXj4reEEXnnfEDazF9oRPGBWOIqVpZyzMNcCkKvsoDQvGfc1vaTNtZckIqDeLLMZ+fFyZkXe04pgojVAaH7Jc5T1G68AzJuFIUKzmDSKhASaGSiMxebmrnA+LUM0Rzjg3aD7KbAnVblAAROlmhiTaBllm03FK5SDaWhJfmrQFY58LbtBDXdpdHl+aiWIiaM5jTeE3poyi/YK3gCJ/af+/QpTI8vyuDcW5bREfLnlhO09znH3kPJiTRST1pQ/rOVEKmVTQRZF9VRpkXpBzsstECaijiTLXhJrfVyRtImYBG2ZkpRGRGJOGSkF7E0wRD8YAFthkukaNAZ6yyG1W30Cs/+43QyabvousoK+HQ8tQlvaRcIsVjLdrmp0zuXe8g/g7Sj5RiVCz4acCjI+t0R6WStffALar3GuSnAvBhZwrrsxV9rV0jHh420VTu0XnR0wHblLHlXtcKxFVCxrLw/zRpLBMb/LADAhjOXcdQjQ8v0tNsjoSuYRjWffdQUyezA+oSSRke+S7Tp4W6kzTr8n6HR18dgD88hadRHGbLpBlMwnBcUpwr5zeUP9GAvpPxeAv9d5FG52cVW1nKn7CUo+h2fRKdPT/+bJCNFZM/JeUndX8S2wTBVssLK29tmzE9iPrsnCVC5zwzjkVez9MlXIPsnDpHU6VBa1mpiYccT156hO6ZOQuznH1mNTjosu/ZbriQkTRV+PO5oHuil0vfCN2TMzQY5aKvGh1w25cculVDdH6+EoBbupeK0ZV6J+CWP7eVP0dECkjGbkIzU9YhJhoYfnRzY1uWCQ1hbpd5mZk8zZ9UUumTNt4c256kyxvfaiN0mLqaG51RUsPmOTrwr/nzsYqJtk9HcyTddHk0R4f1DnLjM51HX77Qi4/oYKLkTzathrFcHN0wh65ZiBEbe35ydumuUAcg2kwUN8mY1Zs/cP/TjHMRuluLvMO1V6jEoSp52Rmhu+fRwfqQz4MizdYcb5Ohc4atosPaIPlDOOIus0VsIgdM7vgDxUmecW2UTUKvd7op4DKzgUaImxHFxrLsI8NM5JTYCGbRQb/AuK7WLFQsavaQ0wj7ZvoubXZV4ASh2zQsYMjRpR7vqAZajtutFPOyP7N+Am6ryAYE5AjWQSHfTtXi6oKR4zMOGwOYdou/NdqpmVKwVVi/8qfKOYLOshgspkcl//jGJEEHs2vxhxk+NyNuZ6ZWNJ6RfMrugAoEQ7UQY2NSXrs0iw5gXAx0aLGEJHf5JQYf3985LaFjy2lX5IpxwbK9At0nll6ZFKVJ17q2VkWLw8nsXmfyrujS5ofF1UfodSTQkcAGf1eQJl310mLtT5jo2Ygm7jWSPEYob98Oi4WoFGvc5+VgAyahC1OXtLbiRO9su0oyJy/ovZV2sJX3sD4WE7sMsdAO5+IXWjBsj3Gdxy0PHFWFSh4oPeuC6b7ypNdinp0byhSmetX1UleyFW9gvN1t20WHTqDt+9RQYtXJkazYcRHN1yR3wGTxJu/Q6H5FIegvQVPWzcp5UTxGoaeWJ+U3E/jLZ2X8TLr9uiK0rVP5bABYl2ZmE2dRVrz+YU59zyQrIfD91N4Hvz/sl5b8BLgtHthHghL1jQlBk+E63RQC2OtJv72bF6oTdEKX/o32GTCaU8uxnmhlsIMi97Sy2GX/dKCOQy9h/MF6frd0yz18U432zFD6QRp8BcMPhe2qTbZdLfPnzYJH3j1uwWHwEblFNXE8BXuXtaqqZhO/tBThxpPEhwLg30lX253OaiGAQYy6X5YdRB5iHZkBpeiUooXKK2igk5Q421N4/ftFNaKfEUxdxdnEPlGMTXFwF3A1u1ZxFsHwiU53i2GGw7/ziW2v9ChWED4JCfjNQu3Z99UIpn+3g6kWo7ua3GpF30IxJrnops7CKzi/fG5u4hk+npxiMmALFTKlyO/zalIRugXhViseBhWCC3ii9FN4Re7xP0IRcw9LFQPPfkYhQCeHTpzZsJ1UPkARRplQ52jDqmWCqEu8ezi9eSzZkeBf6/7P3FmVQbcxu5TrXbfo005Tn0zH43dUbCdqptDIY+YAFitw2Yz0SVfZ7e7U+hO0bd9XoQy6k9E1z7f+6lBIq+WlrQThucmdowP/mka4PHjisFSKytoQRUVURTW7xP2SwvOvKYNuwOafqTu4P82FMaY1ea0l4+ylywOH8WKT5PPi0pvSLlH8GJ2yDbF8sv7dUjOLjqk0xTIM7c8mH6oQxXnjAkNZw9H1MExGVKDmc5PknFUiBwsbRXTKFkNvl2pXsjx8E8qgw63carxfA83GsEIS7iZKeAp0IrQ2CUzJ5R3w5FIRf4TpejzwG51NsoGoDSiLLtoxjqxsJBzE4W6a4Cd8dhPe8tqQgnkZXs+YKRZ7QGwi3gQ2Kp/0KE15dPwTJl3ShOgktiaOauIJNfHMTeKf09oFMCAmVseL/nck3D+OvGuSJi5DyClEt+baNqpn0Ne7RrrsktDGWKvxZRjHJSZpsGnegc2MbcOLyydA2Os6WEhm15OP8pIEN1iG4dxkEjAM2XhGJPT6t6XOrN5UuuGwOOuuKTcy0c5x2khyPAMn9jaC6wYJr8Dqu12ymPaXGtcQ34pu6Ko8ZF1RtTND56qao6tdJVvXP86nTM54+QcpurUQLEmQk5irJidcqVhrnmiO2u1dvunWqGTkk79WRC6zwfwLUwiq5ua9d5dIpiSVEDZZdEKEdtNMR1h+xG1aH/xXuw9NZa22d8JQmfoJzYArhNPJG+UtQsGbNGqIhzpm0Iko8HRR2Wmbff5GYLY/nfajn3A85IvclFVSEQZJalbzqZfGA8cVbVOR8VA452uK6LRDAl2GWds8uiig+psFYlt0F8tun5uKGXQ3gU42Ubo+tTkTooy01IoqopsIdKStY9F3vk8Y9dfrflt9iaDUAroMK8VZbIOqDgWycfCDFwNCQf+3JbMnigsrzMzMf8XMbFDRpdDfiWmUl/dioMJt62VGIbcZmYnGSwZdFGvbljvPv7zLHRI1uVraTUXGihKvK8zxyE5UsfuTDswv9oYvq6w8R/Y3rMuWQldEBazMDi6Pjpf3apDtUyTbeUzVeRgR9vkYWylHINIj735saqnZeznE2cJjxQnJvjDb4W28MWaI1apa+6otUbojDrDjqXXxVi1yKEkX4yh2hbVpem/rfvKVp7e0F45Wnn5K/LDxjiHyrJDPVvqJ5kWFUHFuyUumnj+0FWUqCKezQMcDR2JvWYMiOIWOKtb95Bmd8pWFCo3tI3g4H3mpNh6NHSny2OHSQj/IknuVbSDPnWhUqzDXmJZwj8ts9MQwzomyAW3EhPJeeAHvSumR3E/aKJcj19pZOBjcvbhexuoDDKOdX9DW7ZhYHaJqLSSeKl67xP6zxviVS13usEXTzFqLmxDx8si29txCpFSNBOOp4i1Vr+2AfRTeIyv0kHejcevg8OKi6lHrNX/+rjX7KCanGnAPmViISou3miIzauwSuaotHmBIE3TmmYAq4Sgz2ohdj5rHNOdaGpoXcal118ebAfzp5siRu3+0tG1UnfWEoMQjr9vzsqEBhMpefFbSso/G7L0eI/jcbE3Zoo0QJNGaitZioCvMezUvzOGE0U+NqsgWRmHH2bFmm95XXvux9oVawnJrbX+CDXrRKbNsOabSRlG6E7W+VcBNxPZcOrEO71a9XrVSoz5W8JcyxfkVUEbDtLyksWy1FcnrBktaPda/xS559tDkXsdCW9kgRmK0o+z4vZiyx18ww8CZF5X5ajQV5nJHSmLlZBrFK/rMg6wpLsKgygKC6jY04sFimSCjFuZmdC+5vMnBJQtxGsfUi6tSCa/jxP9owXMj7on9aLK5h9DBWiANTbIoFE7DVBljzefo2+tf3rYpLoBu1gxfuOZnw9uaOessDzPHdHvvtMA8WGN4XWPdAhsVd9FN4AnDwNyIGv4+v4y0ocNbgKt6AfSLdjroOlCaBF6L6AY6itFxR1mjPFtYY014rQXJyzafSjPuoQ8c3XERuqju1BuX/8shcc5Vv936ZVs2ekZMRVZyihNk6ifoeNVj6QpfnSjS2mjJoQY+ekaeVoV5+zhuDtCTmqDjrj/pcv4igk5rLUJHeEaI5L0MdrTqUnQ87VY2GkBckdHGmkta7Gz4UpGJceUhlyIoPEEnToNkdBXAEt+MI39pe2mj3PFjlVWFeffoOK4tnkGHUVTvT2xKGrO5S4225zSI2l3iaZVEGQYeg8PrbHN0IkxO8hA0KpQhcaPv25b5fFdIXZsaY73FmWgWHSqFe92DJpiiJiAfXxFHLAqokI9xvUsFEZJYYll0/L6Cmpd9wxL9hkr3axJM2LxAv5Z+rTM7ReL+qICuA07di5VgdDTk1kbVDmwuWxStxtRAdJF3JoeOTc066IBtxnB3X1rtqiUCGPOTN7qrviH2P2K3ThZdxzfMGhf/wOyKQq2kFlSrBH2CUkslt6rvEMK/kS81hw7Wn5UVMmMcv8ZXK6nj1S7hLTfcWT6valan0ZZZdDWSs9iK4xUiaCuFc951NtVRuCj601ssnj04oN36d1aAHfIVZwTfk0EJ/twSt2HXqz4hgw7Au6CoNOnyuxLvAaZ89RG6qXVRN0f3IsSwpKOhqKWgLZo75mp0a4sSACadVw/KhsEHIR/VY00BJmd+MNaj9SyI5gTrDZ+eirutzD8YX9TKZ1EM24aKN7j9/pB9gFvAb7xT6HxSER/YFfUkE6jTM8dGrGNZxYSvJ6aGCE9iVayFV3J5U4OG/VXAk9SJfviKEiUVh9EZqxwf0dWwpawIzJZa3qNWy6pBfCOBvVLEjYyqdfb8xvftAsxWC8ojQEzr8LPYxIC8hahWbmruvAlAhLbfuKLYQ49u2q3lJEtMus2jMSm6ft2PJJIa8JH1+Gxp/JiPaB+7Opr0awlgcAqsKN1Oo5/L6aA6QpFT6u3uliFOMFW62JdWavkxYgOc7O66CGsgqk6Pu/3af5OaEt0g3V9tD1SLIiJ61iX8PWxLCcCebg1Hja4oVDRLC3anad8vvd0cYQ/63mke6E50eU6XGJaynPzarC0GcLhM+cAgqrpF74v5djn2vOn0NhwOb1PP25+W22twp45mKMkN0ho9hiWF938V4Rrab++WpqZhAERRVEPTNF0Q+5faU8z0nNlkM/my82a/HJogISWWx4ulp6wpJ6IYOlXO4VRGzv4c8ZI26+lpu7hTi01Ao2eaJCaTsdLQHMu9H3fj6cj/XyFLKJaJw/3ptLuej4sD0uJ4notL6/3fnPlZkZ5IzP87rH/0j/7RP/pH/+jX0H9gsJX0V1fSfAAAAABJRU5ErkJggg==",
            xref="x",
            yref="y",
            x=452.5,
            y=47.5,
            sizex=95,
            sizey=95,
            sizing="stretch",
            opacity=1,
            layer="below")
    )
    fig.add_annotation(x=400, y=200, font=dict(size=20, color='DarkGreen'),
                       text="Offensive FG%",
                       showarrow=False,
                       yshift=10)
    fig.add_annotation(x=600, y=200, font=dict(size=20, color='DarkGreen'),
                       text="Defensive FG%",
                       showarrow=False,
                       yshift=10)

    draw_rect(-45, -255, 1045, 255)
    draw_rect(-45, -80, 140, 80)
    draw_rect(860, -80, 1045, 80)
    fig.add_shape(type="line", x0=500, y0=-255, x1=500, y1=-42, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=500, y0=42, x1=500, y1=255, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=-220, x1=91, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=220, x1=91, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=-220, x1=909, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=220, x1=909, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(y_center=0, a=240, b=-238, start_angle=2.26 * np.pi / 6,
                                   end_angle=-2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(x_center=1000, a=-240, b=238, start_angle=-2.26 * np.pi / 6,
                                   end_angle=2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="FG%",
        ),
    )
    return fig


@app.callback(
    Output(component_id='shot_zone_season', component_property='figure'),
    Input(component_id='select_season_per_zone', component_property='value')
)
# Przygotowanie danych i wykresu do wykresu zdobytych punktów za rzut
def update_graph(chosen_season):
    dff = df.copy()
    dff = dff[dff["season"] == chosen_season]

    dff['procentage'] = 0

    def fg_procentage_3(df_def, column):
        dff_count = df_def[df_def[column] == 1]
        count_miss = dff_count[dff_count["outcome"] == "missed"].count()[0]
        count_made = dff_count[dff_count["outcome"] == "made"].count()[0]
        proctenage = 3 * (count_made / (count_made + count_miss))
        df_def['procentage'] = np.where(df_def[column] == 1, proctenage, df_def['procentage'])

    def fg_procentage_2(df_def, column):
        dff_count = df_def[df_def[column] == 1]
        count_miss = dff_count[dff_count["outcome"] == "missed"].count()[0]
        count_made = dff_count[dff_count["outcome"] == "made"].count()[0]
        proctenage = 2 * (count_made / (count_made + count_miss))
        df_def['procentage'] = np.where(df_def[column] == 1, proctenage, df_def['procentage'])

    fg_procentage_3(dff, 'left_corner')
    fg_procentage_3(dff, 'right_corner')
    fg_procentage_3(dff, 'middle_left_3pt')
    fg_procentage_3(dff, 'middle_right_3pt')
    fg_procentage_3(dff, 'middle_3pt')
    fg_procentage_2(dff, 'deep_2pt_left_corner')
    fg_procentage_2(dff, 'deep_2pt_right_corner')
    fg_procentage_2(dff, 'deep_2pt_left_middle')
    fg_procentage_2(dff, 'deep_2pt_right_middle')
    fg_procentage_2(dff, 'deep_2pt_middle')
    fg_procentage_2(dff, 'close_paint')

    fig = go.Figure(data=go.Scatter(
        x=dff['x'],
        y=dff['y'],
        showlegend=False,
        mode='markers',
        marker=dict(
            symbol='hexagon2',
            size=12,
            color=dff['procentage'],
            colorscale='Viridis',
            showscale=True,
            reversescale=True

        ),
        marker_colorbar_title=dict(text='Points per shot')

    ))

    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    def draw_rect(x0, y0, x1, y1):
        fig.add_shape(type="rect",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="DarkGreen", width=3))

    fig.add_layout_image(
        dict(
            source="data:image/png;base64,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",
            xref="x",
            yref="y",
            x=452.5,
            y=47.5,
            sizex=95,
            sizey=95,
            sizing="stretch",
            opacity=1,
            layer="below")
    )
    fig.add_annotation(x=400, y=200, font=dict(size=20, color='DarkGreen'),
                       text="",
                       showarrow=False,
                       yshift=10)
    fig.add_annotation(x=600, y=200, font=dict(size=20, color='DarkGreen'),
                       text="",
                       showarrow=False,
                       yshift=10)

    draw_rect(-45, -255, 1045, 255)
    draw_rect(-45, -80, 140, 80)
    draw_rect(860, -80, 1045, 80)
    fig.add_shape(type="line", x0=500, y0=-255, x1=500, y1=-42, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=500, y0=42, x1=500, y1=255, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=-220, x1=91, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=-45, y0=220, x1=91, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=-220, x1=909, y1=-220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="line", x0=1045, y0=220, x1=909, y1=220, line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(y_center=0, a=240, b=-238, start_angle=2.26 * np.pi / 6,
                                   end_angle=-2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))
    fig.add_shape(type="path",
                  path=ellipse_arc(x_center=1000, a=-240, b=238, start_angle=-2.26 * np.pi / 6,
                                   end_angle=2.26 * np.pi / 6),
                  line=dict(color="DarkGreen", width=3))

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
