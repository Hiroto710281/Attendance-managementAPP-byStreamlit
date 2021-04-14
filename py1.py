from io import StringIO
from itertools import product
import pandas as pd
from mip import BINARY, Model, maximize, xsum
from more_itertools import pairwise, windowed
import streamlit as st

# 画面左側の作成
# ①「休み希望日」のテキストエリア
s = "Name,Day\n佐藤,1\n田中,2\n鈴木,3\n高橋,8"  # デフォルト値
wishs = st.sidebar.text_area("休み希望日", s, height=300)

dfws = pd.read_csv(StringIO(wishs))  # 希望表
dfws["Shift"] = "休"

# 最適化モデルの作成と求解
# 変数表の元（変数の列なし)
staffs = ["佐藤", "田中", "鈴木", "高橋"]  # 看護師リスト
days = range(1, 9)  # 日付リスト
shifts = ["日", "夜", "休"]  # シフトリスト

d = product(staffs, days, shifts)  # 看護師、日付、シフトの組み合わせ
df = pd.DataFrame(d, columns=["Name", "Day", "Shift"])

# モデル作成
m = Model()

# 変数表に、変数の列を追加（列Var）
df["Var"] = m.add_var_tensor((len(df),), "Var", var_type=BINARY)
m.objective = maximize(xsum(dfws.merge(df).Var))  # 目的関数

# 制約条件の追加
for _, gr in df.groupby(["Name", "Day"]):
    m += xsum(gr.Var) == 1  # (1) 看護師と日付の組み合わせごとにシフトは1つ

for _, gr in df.groupby("Day"):
    m += xsum(gr[gr.Shift == "日"].Var) >= 2  # (2) 日付ごとに日勤は2名以上
    m += xsum(gr[gr.Shift == "夜"].Var) >= 1  # (3) 日付ごとに夜勤は1名以上

q = "(Day == @d1 & Shift == '夜') | (Day == @d2 & Shift != '休')"
for _, gr in df.groupby("Name"):
    m += xsum(gr[gr.Shift == "日"].Var) <= 4  # (4) 看護師ごとに、日勤は4日以下
    m += xsum(gr[gr.Shift == "夜"].Var) <= 2  # (5) 看護師ごとに、夜勤は2日以下
    for d1, d2 in pairwise(days):
        m += xsum(gr.query(q).Var) <= 1  # (6) 「夜勤と『翌日が休み以外』」はどちらかだけ
    for dd in windowed(days, 4):
        m += xsum(gr.query("Day in @dd & Shift == '休'").Var) >= 1  # (7) 4連続勤務のうち休みが1日以上

# ソルバーで求解
m.optimize()

# 変数表に結果の列Valを追加
df["Val"] = df.Var.astype(float)

# 結果の整形
res = df[df.Val > 0]  # 有効なシフトの行だけを取得
res = res.pivot_table("Shift", "Name", "Day", "first")  # ピボット表
res = res.style.applymap(lambda s: f"color: {'red' * (s == '休')}")

# 画面右側の作成
# ② 結果サマリの表示領域の作成
f"""
# 看護師スケジューリング
## 実行結果
- ステータス : {m.status}
- 希望をかなえた数 : {m.objective_value}
"""
# ③ スケジュール表の作成
st.dataframe(res)
