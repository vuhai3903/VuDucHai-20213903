import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

# Khởi tạo ứng dụng Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout của ứng dụng
app.layout = html.Div([
    html.H1("Phát hiện tấn công mạng - Thời gian thực", style={'textAlign': 'center'}),
    
    # Thông báo cảnh báo (Alert)
    html.Div(id='alerts', style={'color': 'red', 'font-weight': 'bold', 'textAlign': 'center'}),
    
    # Thống kê tổng quan
    html.Div(id='attack-stats', style={'font-size': '18px', 'textAlign': 'center', 'marginTop': '20px'}),
    
    # Bảng điều khiển (Control Panel) để lọc tấn công
    html.Div([  
        dcc.Dropdown(
            id='attack-filter',
            options=[
                {'label': 'Tất cả', 'value': 'all'},
                {'label': 'DDoS', 'value': 'ddos'},
                {'label': 'Phishing', 'value': 'phishing'}
            ],
            value='all',  # Mặc định lọc tất cả
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Biểu đồ tấn công theo thời gian
    dcc.Graph(id='attack-graph'),
    
    # Biểu đồ pie chart cho các loại tấn công
    dcc.Graph(id='attack-pie-chart', style={'marginTop': '20px'}),
    
    # Thời gian cập nhật biểu đồ
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # cập nhật mỗi 5 giây
        n_intervals=0
    )
])

# Callback để cập nhật biểu đồ và thông báo
@app.callback(
    [Output('attack-graph', 'figure'),
     Output('attack-stats', 'children'),
     Output('alerts', 'children'),
     Output('attack-pie-chart', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('attack-filter', 'value')]
)
def update_graph(n, filter_value):
    try:
        df = pd.read_csv("attack_log.csv", names=["timestamp",  "samples"])


        # Kiểm tra nếu DataFrame không trống
        if df.empty:
            return {}, "Không có dữ liệu tấn công", "Hệ thống ổn định.", {}

        # Chuyển đổi thời gian và xử lý dữ liệu
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['samples'] = pd.to_numeric(df['samples'], errors='coerce')


        # Lấy 10 bản ghi cuối cùng
        df = df.tail(10)

        # Tính toán các thống kê
        total_attacks = df.shape[0]
        max_attack = df['samples'].max()

        # Thông báo cảnh báo
        if max_attack > 1000:
            alert_message = "Cảnh báo: Tấn công DDoS có thể đang xảy ra!"
        else:
            alert_message = "Hệ thống ổn định."

        stats = f"Tổng số cuộc tấn công: {total_attacks} | Mức tấn công cao nhất: {max_attack}"

        # Cập nhật biểu đồ Line chart
        fig_line = px.line(df, x="timestamp", y="samples", title="Tấn công theo thời gian")
        fig_line.update_layout(xaxis_title='Thời gian', yaxis_title='Số lượng mẫu')

        # Cập nhật biểu đồ Pie chart
        fig_pie = px.pie(df, names='samples', title='Tỷ lệ các loại tấn công')

        return fig_line, stats, alert_message, fig_pie
    except Exception as e:
        print(f"Lỗi: {e}")
        return {}, "Không có dữ liệu tấn công", "Hệ thống gặp sự cố.", {}


# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
