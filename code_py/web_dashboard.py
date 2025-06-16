import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests


widget_style = {
    'border': '1px solid #ccc',
    'padding': '10px',
    'borderRadius': '10px',
    'backgroundColor': '#f5f5f5',
    'boxShadow': '2px 2px 6px rgba(0,0,0,0.1)',
}
gray_bg = '#f5f5f5'

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("DASHBOARD GIÁM SÁT DDoS", style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Interval(id='interval-update', interval=10*1000, n_intervals=0),

    html.Div([
        html.Div([
            dcc.Graph(id='line-graph')
        ], style={**widget_style, 'width': '40%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='bar-graph')
        ], style={**widget_style, 'width': '40%', 'display': 'inline-block', 'marginLeft': '4%'})
    ], style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            dcc.Graph(id='dir-graph')
        ], style={**widget_style, 'width': '40%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='proto-graph')
        ], style={**widget_style, 'width': '40%', 'display': 'inline-block', 'marginLeft': '4%'})
    ], style={'textAlign': 'center'})
])

@app.callback(
    Output('line-graph', 'figure'),
    Output('bar-graph', 'figure'),
    Output('dir-graph', 'figure'),
    Output('proto-graph', 'figure'),
    Input('interval-update', 'n_intervals')
)
def update_graphs(n):
    url = "http://127.0.0.1:8000/"
    try:
        response = requests.get(url)
        data = response.json().get("data", [])
    except Exception as e:
        data = []
        print("Lỗi khi gọi API:", e)

    columns = [
        "id", "time", "duration", "count_botnet",
        "tot_pkts", "tot_bytes", "src_bytes",
        "dir_forward", "dir_bidirectional",
        "proto_icmp", "proto_tcp", "proto_udp",
        "dTos_0_0", "dTos_10_0"
    ]
    df = pd.DataFrame(data, columns=columns)


    def fig_line(): # đồ thị đường biểu hiện số mẫu DDoS
        fig = px.line( df, x='time', y='count_botnet', color_discrete_sequence=['red']  ,
            title='Số lượng DDoS theo Thời gian',
            labels={'time': 'Thời gian', 'count_botnet': 'Số lượng DDoS'},
            hover_data={'time': True, 'count_botnet': True, 'duration': True}
        )
        return fig

    def fig_bar():  # đồ thị cột biểu hiện cuộc tấn công trong 1 ngày
        df["time"] = pd.to_datetime(df["time"])
        df['date'] = df['time'].dt.date
        df_attack_count = df.groupby('date').size().reset_index(name='attacks_in_day')
        df_attack_count.columns = ['date', 'attacks_in_day']
        
        latest_date = df_attack_count['date'].max()
        attacks_counts = df_attack_count[df_attack_count['date'] == latest_date]['attacks_in_day'].values[0]
        fig = px.bar( df_attack_count, x='date', y='attacks_in_day', title=f'Cuộc tấn công gần nhất vào ({latest_date}) có : {attacks_counts}',
            labels={'date': 'Ngày', 'attacks_in_day': 'Số cuộc tấn công'}
        ) 
     
        return fig

    def fig_dir():  # đồ thị tròn mô tả tỉ lệ hướng đi của cuộc tấn công 
        dir_data = df[['dir_forward', 'dir_bidirectional']].sum().reset_index()
        dir_data.columns = ['Direction', 'Count']
        fig = px.pie(dir_data, names='Direction',values='Count',title='Tỷ lệ Hướng Truyền ')
        
        return fig

    def fig_proto(): # đồ thị tròn mô tả tỉ lệ giao thức cuộc tấn công
      
        proto_data = df[['proto_icmp', 'proto_tcp', 'proto_udp']].sum().reset_index()
        proto_data.columns = ['Protocol', 'Count']
        fig = px.pie( proto_data,names='Protocol',values='Count',title='Tỷ lệ Giao Thức ' )
        
        return fig

    figs = [fig_line(), fig_bar(), fig_dir(), fig_proto()]
    for fig in figs:
        fig.update_layout(
            plot_bgcolor=gray_bg,
            paper_bgcolor=gray_bg,
            font=dict(color='black'),
            margin=dict(t=30, l=30, r=30, b=30)
        )
    return figs

if __name__ == '__main__':
    app.run(debug=True)
