#INFO: plotlyでよく使うレイアウトの設定，保存，描画などの関数をまとめておく．
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List


pio.templates.default = "plotly_white"

## version取得関数
def version():
    return "0.0.1"

def save_and_plot(fig, filename:str='NoName', path:pathlib='./', format:List[str]=['svg', 'png']):
    for f in format:
        pio.write_image(fig, path/f'{filename}.{f}')
    fig.show()

def update_layout(fig, layout:Dict = pio.templates[pio.templates.default]):
    fig.update_layout(layout)

def dark_mode():
    pio.templates.default = "plotly_dark"

def customize_layout():
    costum_template = dict(
        layout=go.Layout(
            height=400,
            width=400,
            plot_bgcolor='#E6E6E6',
            font=dict(family="PlemolJP", size=16, color="#585858"),
            title=dict(font=dict(size=26, color="#585858"), x=0.5, y=0.97),
            legend=dict(xanchor='left', yanchor='bottom', font=dict(size=16, color="#585858")),
            margin=dict(l=60, r=20, t=50, b=40),
            xaxis=dict(linewidth=1, linecolor="#424242", color="#424242", title_standoff=0.01),
            yaxis=dict(linewidth=1, linecolor="#424242", color="#424242", title_standoff=0.01),
        )
    )

    # テンプレを適用
    pio.templates["costum_template"] = costum_template
    pio.templates.default = "costum_template"

def thesis_layout():
    my_thesis_template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="Times New Roman",
                size=14,
                color="black"
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',

            # タイトル設定: 位置・フォント・パディング
            title=dict(
                text="Figure Title Example",
                # タイトルを中央配置したい場合
                x=0.5,
                y=0.95,
                xanchor='center',  # x=0.5の基準を文字列中央に
                yanchor='top',     # y=0.95の基準を文字列上部に
                font=dict(
                    size=20,
                    color='black'
                ),
                # タイトル周りの余白を追加したいときは pad も使える
                # pad=dict(t=10, b=10)
            ),
            
            # --- x軸周りの設定 ---
            xaxis=dict(
                showline=True,          # 軸線を表示
                linewidth=1,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray',
                
                # 軸ラベル(タイトル)と軸の間のスペースを指定
                # 値を大きくするとラベルが下にずれて枠と被りにくくなる
                title_standoff=5,
                
                # ゲージ(目盛り)の外側表示と長さなど
                ticks='outside',
                ticklen=5,
                tickwidth=1,
                tickcolor='black',
                
                # ラベルがはみ出す場合に余白を自動調整
                automargin=True
            ),
            
            # --- y軸周りの設定 ---
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray',
                title_standoff=10,  # 必要に応じて調整
                ticks='outside',
                ticklen=5,
                tickwidth=1,
                tickcolor='black',
                automargin=True
            ),
            
            # 図全体の余白(左, 右, 上, 下)
            margin=dict(l=80, r=20, t=50, b=50),

        )
    )

    # テンプレを適用
    pio.templates["my_thesis_template"] = my_thesis_template
    pio.templates.default = "my_thesis_template"