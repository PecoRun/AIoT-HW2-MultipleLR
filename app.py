"""Streamlit App：Human-AI Interaction 信任分數預測（多元線性回歸）

執行：streamlit run app.py
模型由 HW2_MultipleLR.ipynb 訓練後存成 models/model.joblib。
"""
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
DATA_PATH = Path(__file__).parent / "data" / "Human-AI_Interaction.csv"

st.set_page_config(page_title="Multiple Linear Regression: Human-AI Interaction", layout="wide")


@st.cache_resource
def load_model(mtime: float):
    """mtime 只用來當快取鍵：模型檔一更新，就會重新載入。"""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data(mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def predict_with_intervals(m, x: pd.Series, alpha=0.05):
    """回傳 (預測值, CI 下界, CI 上界, PI 下界, PI 上界)。"""
    x = x.reindex(m["params"].index)
    mean = float(x @ m["params"])
    var_mean = float(x.values @ m["cov"].values @ x.values)
    t = stats.t.ppf(1 - alpha / 2, m["df_resid"])
    ci = t * np.sqrt(var_mean)
    pi = t * np.sqrt(var_mean + m["scale"])
    return mean, mean - ci, mean + ci, mean - pi, mean + pi


RMSE_COLORS = {"Train RMSE": "#9ecae9", "CV RMSE": "#4C78A8", "Test RMSE": "#F58518"}


def rmse_dot_chart(df: pd.DataFrame, labels: pd.Series, show: list[str] | None = None) -> alt.Chart:
    """Train / CV / Test RMSE 的互動點圖。show 指定要顯示的指標（預設全部），橫軸範圍固定，切換時不會跳動。"""
    cols = list(RMSE_COLORS)
    show = show or cols
    wide = df[cols].copy()
    wide["標籤"] = labels.values
    order = list(wide["標籤"])
    spans = pd.DataFrame({"標籤": wide["標籤"], "最小": wide[show].min(axis=1), "最大": wide[show].max(axis=1)})
    long = wide.melt(id_vars="標籤", var_name="指標", value_name="RMSE")
    long = long[long["指標"].isin(show)]

    lo, hi = float(wide[cols].min().min()), float(wide[cols].max().max())
    pad = (hi - lo) * 0.08 or 0.01
    y = alt.Y("標籤:N", sort=order, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12))
    x_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False)
    line = alt.Chart(spans).mark_rule(color="#888888", strokeWidth=3).encode(
        y=y, x=alt.X("最小:Q", scale=x_scale, title="RMSE（越小越好）", axis=alt.Axis(tickCount=8, format=".3f")), x2="最大:Q")
    dots = (
        alt.Chart(long)
        .mark_point(filled=True, size=170, stroke="white", strokeWidth=1)
        .encode(
            y=y, x=alt.X("RMSE:Q", scale=x_scale, axis=alt.Axis(tickCount=8, format=".3f")),
            color=alt.Color("指標:N", scale=alt.Scale(domain=cols, range=list(RMSE_COLORS.values())), legend=alt.Legend(orient="top", title=None, symbolType="square", symbolSize=180, labelFontSize=12, padding=8)),
            tooltip=["標籤", "指標", alt.Tooltip("RMSE:Q", format=".4f")],
        )
    )
    return (line + dots).properties(height=alt.Step(34))  # 每列固定 34px，列數再多也不會擠在一起


def rmse_dot_panel(df: pd.DataFrame, labels: pd.Series, key: str) -> None:
    """RMSE 點圖 + 指標選擇（全部 / Train / CV / Test）。"""
    choice = st.radio("顯示指標", ["全部", *RMSE_COLORS], horizontal=True, key=key)
    st.altair_chart(rmse_dot_chart(df, labels, None if choice == "全部" else [choice]), width="stretch")


if not MODEL_PATH.exists():
    st.error("找不到 models/model.joblib，請先執行 HW2_MultipleLR.ipynb 訓練並儲存模型。")
    st.stop()

m = load_model(MODEL_PATH.stat().st_mtime)
cat_used = sorted({raw for raw, _ in m["dummy_map"].values()})

st.title("Multiple Linear Regression: Human-AI Interaction")
st.caption("多元線性回歸（Multiple Linear Regression）· 資料集：Kaggle Human-AI Interaction")

st.sidebar.header("輸入特徵")
values = {}
for name in m["num_features"]:
    lo, hi, med = m["num_range"][name]
    if float(lo).is_integer() and float(hi).is_integer():
        values[name] = float(st.sidebar.slider(name, int(lo), int(hi), int(round(med))))
    else:
        values[name] = st.sidebar.slider(name, float(round(lo, 3)), float(round(hi, 3)), float(round(med, 3)))

choices = {}
for raw in cat_used:
    choices[raw] = st.sidebar.selectbox(raw, m["cat_levels"][raw], index=len(m["cat_levels"][raw]) - 1)

x = pd.Series(0.0, index=m["params"].index)
x["const"] = 1.0
for k, v in values.items():
    x[k] = v
for col, (raw, level) in m["dummy_map"].items():
    x[col] = 1.0 if choices[raw] == level else 0.0

mean, ci_lo, ci_hi, pi_lo, pi_hi = predict_with_intervals(m, x)
lo_b, hi_b = m["target_range"]

c1, c2, c3 = st.columns(3)
c1.metric("預測信任分數", f"{mean:.1f}")
c2.metric("95% 信賴區間 (CI)", f"{ci_lo:.1f} ~ {ci_hi:.1f}", help="在這組特徵下，「平均」信任分數的範圍")
c3.metric("95% 預測區間 (PI)", f"{pi_lo:.1f} ~ {pi_hi:.1f}", help="單一新使用者實際信任分數可能落入的範圍")

PI_LABEL, CI_LABEL = "95% 預測區間 (PI)", "95% 信賴區間 (CI)"
COLORS = alt.Scale(domain=[PI_LABEL, CI_LABEL], range=["#F58518", "#4C78A8"])
x_scale = alt.Scale(domain=[min(lo_b, pi_lo) - 2, max(105, pi_hi + 2)], clamp=True)

bands = pd.DataFrame({"區間": [PI_LABEL, CI_LABEL], "下界": [pi_lo, ci_lo], "上界": [pi_hi, ci_hi]})
bar_chart = (
    alt.Chart(bands)
    .mark_bar(height=28)
    .encode(
        x=alt.X("下界:Q", scale=x_scale, title="latent_trust_score"),
        x2="上界:Q",
        y=alt.Y("區間:N", title=None, sort=[PI_LABEL, CI_LABEL]),
        color=alt.Color("區間:N", scale=COLORS, legend=None),
        tooltip=["區間", alt.Tooltip("下界:Q", format=".2f"), alt.Tooltip("上界:Q", format=".2f")],
    )
)
rule = (
    alt.Chart(pd.DataFrame({"預測值": [mean]}))
    .mark_rule(color="#00A3FF", strokeWidth=3)
    .encode(x=alt.X("預測值:Q", scale=x_scale), tooltip=[alt.Tooltip("預測值:Q", format=".2f")])
)
st.altair_chart((bar_chart + rule).properties(height=130), width="stretch")
st.caption("將滑鼠移到區間或預測線上可看到精確數值。")

if pi_hi > 100 or mean > 100:
    st.info("資料中的信任分數上限為 100；線性模型的區間可能超出此範圍，解讀時請以 0~100 為界。")

st.subheader("特徵敏感度分析")
focus = st.selectbox("選擇要變動的特徵（其餘特徵維持左側輸入值）", m["num_features"], index=m["num_features"].index("trust_history") if "trust_history" in m["num_features"] else 0)
f_lo, f_hi, _ = m["num_range"][focus]
grid = np.linspace(f_lo, f_hi, 100)
Xg = np.tile(x.values, (len(grid), 1))
Xg[:, list(x.index).index(focus)] = grid
cov = m["cov"].values
mean_g = Xg @ m["params"].values
var_g = np.einsum("ij,jk,ik->i", Xg, cov, Xg)
t_crit = stats.t.ppf(0.975, m["df_resid"])
sens = pd.DataFrame({
    "特徵值": grid, "預測值": mean_g,
    "CI 下界": mean_g - t_crit * np.sqrt(var_g), "CI 上界": mean_g + t_crit * np.sqrt(var_g),
    "PI 下界": mean_g - t_crit * np.sqrt(var_g + m["scale"]), "PI 上界": mean_g + t_crit * np.sqrt(var_g + m["scale"]),
})
xs = alt.X("特徵值:Q", title=focus)
pi_area = alt.Chart(sens).mark_area(color="#F58518", opacity=0.25).encode(xs, y=alt.Y("PI 下界:Q", title="latent_trust_score", scale=alt.Scale(zero=False)), y2="PI 上界:Q")
ci_area = alt.Chart(sens).mark_area(color="#4C78A8", opacity=0.6).encode(xs, y="CI 下界:Q", y2="CI 上界:Q")
line = alt.Chart(sens).mark_line(color="#00A3FF", strokeWidth=2).encode(xs, y="預測值:Q")
hover = alt.Chart(sens).mark_point(opacity=0, size=60).encode(
    xs, y="預測值:Q",
    tooltip=[alt.Tooltip(c, format=".2f") for c in ["特徵值", "預測值", "CI 下界", "CI 上界", "PI 下界", "PI 上界"]],
)
cur = alt.Chart(pd.DataFrame({"特徵值": [values[focus]], "預測值": [mean]})).mark_point(color="#E45756", size=120, filled=True).encode(xs, y="預測值:Q", tooltip=["特徵值", "預測值"])
st.altair_chart((pi_area + ci_area + line + hover + cur).properties(height=340).interactive(), width="stretch")
st.caption("橘色為 95% 預測區間 (PI)、藍色為 95% 信賴區間 (CI)、紅點為目前輸入值。可用滾輪縮放、拖曳平移、雙擊還原。")


# ---------------------------------------------------------------------------
# 下方各面板：CRISP-DM、資料預覽、Feature Selection、係數、模型比較、Evaluation、預測區間與假設診斷
# ---------------------------------------------------------------------------
res = m["results"]
final_test = res.loc["特徵選擇後 - Test"]
tp = m["test_pred"]
coverage = float(((tp["actual"] >= tp["obs_ci_lower"]) & (tp["actual"] <= tp["obs_ci_upper"])).mean())
n_train, n_test = m["split_sizes"]
n_raw, n_clean = m["n_raw"]
comp, sel = m["comparison"], m["selection"]

st.divider()
st.header("分析報告")

st.subheader("1. CRISP-DM 六階段")
n_final = int(final_test["特徵數"])
cv_gap = comp["CV RMSE"].max() / comp["CV RMSE"].min() - 1
crisp = [
    ("1. Business Understanding", "商業理解", f"""
- **問題**：哪些因素會影響使用者對 AI 的信任分數？
- **目標**：用多元線性回歸預測 `latent_trust_score`，找出關鍵影響因素
- **成果**：預測模型 + 95% CI / PI + Streamlit 預測工具
"""),
    ("2. Data Understanding", "資料理解", f"""
- Kaggle Human-AI Interaction：**{n_raw:,} 筆 × 21 欄**（20 特徵 + 目標）
- 每欄約 **2%** 缺失、另有重複列
- 與目標最相關：`trust_history`（+0.48）、`bias_level`（−0.32）
- `model_accuracy` 與 `error_rate` 相關 **−0.97**（高度共線）
"""),
    ("3. Data Preparation", "資料準備", f"""
- 移除重複與目標缺失：{n_raw:,} → **{n_clean:,}** 筆
- 80/20 切分（**{n_train:,}** / **{n_test:,}**），以訓練集中位數 / 眾數插補
- One-Hot Encoding → 30 個特徵
- VIF > 16 → 移除 `error_rate`，最大 VIF 降為 1.77
"""),
    ("4. Modeling", "建模", f"""
- **七種**特徵選擇方法，核心特徵一致
- **最終模型**：OLS + **{n_final} 個特徵**（可得係數檢定與區間）
- 對照 Ridge / Lasso / ElasticNet 共 12 組，CV RMSE 差距僅 {cv_gap:.2%}，正則化沒有實質幫助
"""),
    ("5. Evaluation", "評估", f"""
- 測試集 **R² = {final_test['R²']:.3f}**、Adj R² = {final_test['Adj R²']:.3f}
- RMSE = {final_test['RMSE']:.2f}、MAE = {final_test['MAE']:.2f}
- 95% PI 涵蓋率 **{coverage:.1%}**
- 五大假設：四項通過，Jarque-Bera 輕微未通過（偏離極小）
"""),
    ("6. Deployment", "部署", """
- 本頁即為 Deployment：左側輸入特徵，即時預測並顯示 95% CI / PI
- 模型存於 `models/model.joblib`，以 `√(xᵀ·Cov·x + σ²)` 計算區間，無需重新訓練
- 完整分析見 `HW2_MultipleLR.ipynb`
"""),
]
for row in (crisp[:3], crisp[3:]):
    cols = st.columns(3)
    for col, (title, zh, body) in zip(cols, row):
        with col.container(border=True, height=270):
            st.markdown(f"**{title}**")
            st.caption(zh)
            st.markdown(body)

st.subheader("2. 資料預覽")
with st.container():
    raw = load_data(DATA_PATH.stat().st_mtime) if DATA_PATH.exists() else None
    if raw is None:
        st.info("找不到 data/Human-AI_Interaction.csv。")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("筆數", f"{len(raw):,}")
        d2.metric("欄位數", raw.shape[1])
        d3.metric("缺失值比例", f"{raw.isna().mean().mean():.1%}")
        d4.metric("目標變數平均", f"{raw[m['target']].mean():.1f}")
        key_cols = ["trust_history", "model_accuracy", "bias_level", "feedback_quality", "cognitive_load", "ui_complexity", m["target"]]
        st.dataframe(raw[[c for c in key_cols if c in raw.columns]].head(10), hide_index=True)
        st.caption(f"僅顯示前 10 筆與重要欄位（最終模型的主要特徵 + 目標變數 `{m['target']}`），完整資料共 {raw.shape[1]} 欄。")

st.subheader("3. Feature Selection：七種方法的 RMSE 比較")
with st.container():
    st.write(
        f"七種方法選出的特徵子集，CV RMSE 最大差距僅約 {(sel['CV RMSE'].max() / sel['CV RMSE'].min() - 1):.2%}，"
        "預測力沒有實質差異；Train 與 CV 非常接近，沒有過度擬合。"
    )
    st.dataframe(sel.round(4), hide_index=True, height=35 * (len(sel) + 1) + 3)
    rmse_dot_panel(sel, sel["方法"] + "（" + sel["特徵數"].astype(str) + " 個）", key="sel_rmse_metric")
    st.caption("Backward、RFECV、RFE、SFS 選出完全相同的 9 個特徵，所以四列的點完全重疊。選「全部」可同時看三種 RMSE；橫軸不從 0 開始。")

st.subheader("4. 模型係數")
with st.container():
    coef = pd.DataFrame({"係數": m["params"], "標準誤": np.sqrt(np.diag(m["cov"].values))}, index=m["params"].index)
    st.dataframe(coef.round(4))

st.subheader("5. 模型比較：OLS vs. Ridge vs. Lasso vs. ElasticNet")
with st.container():
    st.write(
        f"12 組模型的 5-fold CV RMSE 最大差距僅約 {(comp['CV RMSE'].max() / comp['CV RMSE'].min() - 1):.2%}，"
        "正則化沒有實質幫助，因此採用最精簡、能提供可靠區間的 OLS + 9 特徵。"
    )
    comp_sorted = comp.sort_values("CV RMSE").reset_index(drop=True)   # 以 CV RMSE（模型選擇準則）由小到大排序
    num_cols = comp_sorted.select_dtypes("number").columns
    highlight_best = lambda row: ["background-color: rgba(46, 160, 67, 0.40); font-weight: 600" if row.name == 0 else "" for _ in row]
    st.dataframe(
        comp_sorted.style.apply(highlight_best, axis=1).format({c: "{:.4f}" for c in num_cols}),
        hide_index=True, height=35 * (len(comp_sorted) + 1) + 3,
    )
    best = comp_sorted.iloc[0]
    st.caption(
        f"依 CV RMSE 由小到大排序，綠底為最佳：{best['特徵集']} · {best['模型']}（CV RMSE = {best['CV RMSE']:.4f}）。"
        "但它與最終採用的 OLS + 9 特徵只差約 0.07%，實務上無差別，因此仍以簡潔、可解釋的 OLS + 9 特徵為最終模型。"
    )
    labels = comp["特徵集"].str.split("（").str[0] + " · " + comp["模型"]
    rmse_dot_panel(comp, labels, key="comp_rmse_metric")
    st.caption("每列的三個點分別是 Train、CV、Test RMSE；可用上方選項切換顯示的指標（選「全部」恢復），滑鼠移上去看數值。橫軸不從 0 開始，請留意刻度範圍。")

st.subheader("6. Evaluation：效能指標 (MAE / MSE / RMSE / R² / Adjusted R²)")
with st.container():
    st.markdown("**最終模型（9 個特徵）· 測試集**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{final_test['MAE']:.3f}")
    c2.metric("MSE", f"{final_test['MSE']:.3f}")
    c3.metric("RMSE", f"{final_test['RMSE']:.3f}")
    c4.metric("R²", f"{final_test['R²']:.4f}")
    c5.metric("Adjusted R²", f"{final_test['Adj R²']:.4f}")
    st.markdown("**全特徵 vs. 特徵選擇後**")
    st.dataframe(res[["MAE", "MSE", "RMSE", "R²", "Adj R²", "特徵數", "5-fold CV RMSE"]].round(4))
    st.caption(
        "9 個特徵的表現與 30 個特徵幾乎相同，Adjusted R² 還略高；Train 與 Test 接近，沒有明顯過度擬合。"
        "5-fold CV RMSE 僅在訓練集上計算。"
    )

    st.markdown("**實際值 vs. 預測值（測試集）**")
    lim = [float(min(tp["mean"].min(), tp["actual"].min())) - 1, float(max(tp["mean"].max(), tp["actual"].max())) + 1]
    sc_scale = alt.Scale(domain=lim, zero=False)
    sc_pts = alt.Chart(tp).mark_circle(size=14, opacity=0.35, color="#4C78A8").encode(
        x=alt.X("mean:Q", title="預測值 (Predicted)", scale=sc_scale),
        y=alt.Y("actual:Q", title="實際值 (Actual)", scale=sc_scale),
        tooltip=[alt.Tooltip("actual:Q", title="實際值", format=".2f"), alt.Tooltip("mean:Q", title="預測值", format=".2f")],
    )
    sc_diag = alt.Chart(pd.DataFrame({"v": lim})).mark_line(color="#E45756", strokeDash=[6, 4], strokeWidth=2).encode(x="v:Q", y="v:Q")
    st.altair_chart((sc_pts + sc_diag).properties(height=420).interactive(), width="stretch")
    st.caption(
        f"共 {len(tp):,} 筆測試資料。紅色虛線為 y = x（完美預測），點越貼近虛線預測越準；"
        f"R² = {final_test['R²']:.3f}，RMSE = {final_test['RMSE']:.2f}。可縮放、平移，滑鼠移到點上看數值。"
    )

st.subheader("7. 預測區間與迴歸假設診斷")
with st.container():

    st.markdown("#### 7.1 測試集預測區間 (CI / PI)")
    with st.container():
        st.write(
            f"測試集 {len(tp):,} 筆中，有 **{coverage:.1%}** 的實際值落在 95% 預測區間內（理想約 95%），區間校準良好。"
            "預測區間比信賴區間寬很多，因為它還包含單一觀測值的隨機誤差。"
        )
        sub = tp.iloc[np.linspace(0, len(tp) - 1, 300).astype(int)].reset_index(drop=True)  # 依預測值排序後等距取 300 點
        sub["序號"] = np.arange(len(sub))
        sx = alt.X("序號:Q", title="測試樣本（依預測值由小到大，等距抽樣 300 筆）")
        a_pi = alt.Chart(sub).mark_area(color="#F58518", opacity=0.25).encode(sx, y=alt.Y("obs_ci_lower:Q", title="latent_trust_score", scale=alt.Scale(zero=False)), y2="obs_ci_upper:Q")
        a_ci = alt.Chart(sub).mark_area(color="#4C78A8", opacity=0.7).encode(sx, y="mean_ci_lower:Q", y2="mean_ci_upper:Q")
        a_line = alt.Chart(sub).mark_line(color="#00A3FF", strokeWidth=2).encode(sx, y="mean:Q")
        a_pts = alt.Chart(sub).mark_circle(color="#E45756", size=22, opacity=0.8).encode(
            sx, y="actual:Q",
            tooltip=[alt.Tooltip("actual:Q", title="實際值", format=".2f"), alt.Tooltip("mean:Q", title="預測值", format=".2f"),
                     alt.Tooltip("mean_ci_lower:Q", title="CI 下界", format=".2f"), alt.Tooltip("mean_ci_upper:Q", title="CI 上界", format=".2f"),
                     alt.Tooltip("obs_ci_lower:Q", title="PI 下界", format=".2f"), alt.Tooltip("obs_ci_upper:Q", title="PI 上界", format=".2f")],
        )
        st.altair_chart((a_pi + a_ci + a_line + a_pts).properties(height=380).interactive(), width="stretch")
        st.caption("橘色：95% 預測區間 (PI)；藍色：95% 信賴區間 (CI，很窄)；亮藍線：預測值；紅點：實際值。可縮放、平移，滑鼠移到紅點看數值。")

    st.markdown("#### 7.2 五大回歸假設診斷")
    with st.container():
        chk = m["checks"].copy()
        chk["是否通過"] = chk["是否通過"].map({True: "✅ 通過", False: "⚠️ 未通過"})
        st.dataframe(chk, hide_index=True)
        st.caption(
            f"Jarque-Bera 未通過，但殘差偏態僅 {m['resid_skew']:.2f}、超額峰度僅 {m['resid_kurt']:.2f}，偏離極小；"
            "樣本數近 8,000 筆時，常態性檢定對微小偏離也會顯著，實務上可接受。"
        )

        resid, fitted = np.asarray(m["resid"]), np.asarray(m["fitted"])
        rng = np.random.default_rng(42)
        idx = rng.choice(len(resid), size=min(3000, len(resid)), replace=False)  # 取樣 3000 點，圖面較清楚
        d_res = pd.DataFrame({"Fitted": fitted[idx], "Residual": resid[idx]})
        p1 = alt.Chart(d_res).mark_circle(size=12, opacity=0.35, color="#4C78A8").encode(
            x=alt.X("Fitted:Q", scale=alt.Scale(zero=False)), y="Residual:Q",
            tooltip=[alt.Tooltip("Fitted:Q", format=".2f"), alt.Tooltip("Residual:Q", format=".2f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red").encode(y="y:Q")
        c_a, c_b = st.columns(2)
        c_a.markdown("**殘差 vs 預測值**（線性 / 同質變異）")
        c_a.altair_chart((p1 + zero).properties(height=300).interactive(), width="stretch")

        th, sample = stats.probplot(resid, dist="norm")[0]
        q_idx = np.linspace(0, len(th) - 1, 400).astype(int)
        d_qq = pd.DataFrame({"理論分位數": th[q_idx], "樣本分位數": sample[q_idx]})
        slope, intercept = np.polyfit(th, sample, 1)
        p2 = alt.Chart(d_qq).mark_circle(size=14, color="#4C78A8").encode(
            x="理論分位數:Q", y=alt.Y("樣本分位數:Q", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("理論分位數:Q", format=".2f"), alt.Tooltip("樣本分位數:Q", format=".2f")])
        ref = alt.Chart(pd.DataFrame({"x": [th.min(), th.max()]})).transform_calculate(y=f"{slope}*datum.x+{intercept}").mark_line(color="red").encode(x="x:Q", y="y:Q")
        c_b.markdown("**殘差 Q-Q Plot**（常態性）")
        c_b.altair_chart((p2 + ref).properties(height=300), width="stretch")

        counts, edges = np.histogram(resid, bins=40)
        d_h = pd.DataFrame({"左": edges[:-1], "右": edges[1:], "筆數": counts})
        p3 = alt.Chart(d_h).mark_bar(color="#4C78A8").encode(
            x=alt.X("左:Q", title="Residual"), x2="右:Q", y=alt.Y("筆數:Q"),
            tooltip=[alt.Tooltip("左:Q", format=".2f"), alt.Tooltip("右:Q", format=".2f"), "筆數"])
        st.markdown("**殘差直方圖**")
        st.altair_chart(p3.properties(height=220), width="stretch")
