# corr_matrix.py — 학습 데이터 기간 추천 탭용: 평균기온 기간 적합도 + 요인별 상관도 매트릭스
# app.py 와 같은 폴더에 두고 사용합니다.
import numpy as np
import pandas as pd
import requests
import streamlit as st
from io import StringIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ── 요인 데이터 구글시트 (상관도 분석용) ──
FACTOR_SHEET_ID = "1DQ9tpJA3QSKLhRfbS_DVYQTxNb3xYvD-UbGC1ofnCiA"
FACTOR_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{FACTOR_SHEET_ID}/export?format=csv&gid=0"

# (표준 컬럼명, 시트 헤더에 포함된 키워드)
_COL_KEYS = [
    ("총공급량", "공급량"), ("유효일수", "유효일수"),
    ("평균기온", "평균기온"), ("최저기온", "최저기온"), ("최고기온", "최고기온"),
    ("체감온도", "체감온도"),
    ("총인구수", "총인구수"), ("세대수", "세대수"), ("인구순이동", "순이동"),
    ("고령인구수", "고령인구"), ("소비자물가지수", "물가"), ("청구전", "청구전"),
]

FACTOR_ORDER = ["공급량", "평균기온", "최저기온", "최고기온", "체감온도", "HDD18",
                "유효일수", "총인구수", "세대수", "고령인구수", "인구순이동",
                "소비자물가지수", "청구전"]


@st.cache_data(ttl=1800)
def load_factor_sheet():
    """요인 시트 로드 → (DataFrame[연, 월, 각 요인], 에러메시지)"""
    try:
        resp = requests.get(FACTOR_SHEET_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return None, f"요인 시트 로드 실패: {e}"
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next((c for c in raw.columns if c in ("날짜", "일자", "date", "Date")), raw.columns[0])
    out = pd.DataFrame({"날짜": pd.to_datetime(raw[date_col], errors="coerce")})
    for std, key in _COL_KEYS:
        col = next((c for c in raw.columns if key in c), None)
        if col is not None:
            out[std] = pd.to_numeric(
                raw[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
    out = out.dropna(subset=["날짜"])
    out["연"] = out["날짜"].dt.year.astype(int)
    out["월"] = out["날짜"].dt.month.astype(int)
    out = out.sort_values("날짜").drop_duplicates(["연", "월"]).reset_index(drop=True)
    if "평균기온" in out.columns:
        out["HDD18"] = (18.0 - out["평균기온"]).clip(lower=0)   # 난방도일 성격의 파생 요인
    return out, None


# ══════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════
def _poly3_r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 8:
        return np.nan
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X = poly.fit_transform(x.reshape(-1, 1))
    return float(LinearRegression().fit(X, y).score(X, y))


def _heatmap(corr, height=520):
    labels = list(corr.columns)
    z = corr.values
    text = [[("" if np.isnan(v) else f"{v:.2f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} × %{x}<br>상관계수 %{z:.3f}<extra></extra>",
        colorbar=dict(title="r")))
    fig.update_layout(
        height=height, margin=dict(t=20, b=40, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="bottom", tickangle=-35)
    return fig


# ══════════════════════════════════════════════
# A. 최근 N년 평균기온 적합도 (학습 기간 추천 포맷)
# ══════════════════════════════════════════════
def build_recent_avg_sample(merged, temp_monthly, product, end_year, max_n=6):
    """각 (연, 월)에 대해 '직전 N년 동월 평균기온'을 만든 표본. 모든 N이 같은 표본을 쓰도록 맞춤."""
    tm = dict(zip(zip(temp_monthly["연"].astype(int), temp_monthly["월"].astype(int)),
                  temp_monthly["월평균기온"].astype(float)))
    base = merged[merged["연"] <= end_year][["연", "월", "월평균기온", product]].dropna().copy()
    min_ty = int(temp_monthly["연"].min())
    sample, nmax = None, 0
    for nmax in range(max_n, 0, -1):
        cand = base[base["연"] - nmax >= min_ty]
        if len(cand) >= 24:
            sample = cand.copy()
            break
    if sample is None:
        return None, 0
    for n in range(1, nmax + 1):
        vals = []
        for y, m in zip(sample["연"], sample["월"]):
            ts = [tm.get((int(y) - k, int(m))) for k in range(1, n + 1)]
            vals.append(np.nan if any(t is None for t in ts) else float(np.mean(ts)))
        sample[f"{n}년평균"] = vals
    sample = sample.rename(columns={product: "공급량", "월평균기온": "당해 실제기온"})
    return sample.dropna().reset_index(drop=True), nmax


def render_recent_temp_fit(merged, temp_monthly, product, end_year, render_table, max_n=6):
    """'학습 기간 추천' 바로 아래: 최근 N년 평균기온을 쓰면 적합도가 좋은가?
    render_table: app.py 의 _render_highlight_table 을 그대로 넘겨 같은 포맷으로 표시."""
    st.markdown('<div class="sub">🌡️ 최근 N년 평균기온 적합도 추천</div>', unsafe_allow_html=True)
    st.caption(f"각 월의 기온을 '직전 N년 동월 평균기온'으로 대체했을 때, 공급량과의 Poly-3 적합도(R²)를 비교 · "
               f"{product} · {end_year}년까지의 실적 사용")

    sample, nmax = build_recent_avg_sample(merged, temp_monthly, product, end_year, max_n)
    if sample is None or nmax < 1:
        st.warning("최근 N년 평균기온 적합도를 계산할 데이터가 부족합니다.")
        return None

    y = sample["공급량"].values.astype(float)
    r2_actual = _poly3_r2(sample["당해 실제기온"].values, y)
    rows = []
    for n in range(1, nmax + 1):
        x = sample[f"{n}년평균"].values.astype(float)
        r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"기간": f"직전 {n}년", "추천연도": f"최근 {n}년 평균기온",
                     "R2": _poly3_r2(x, y), "상관계수(r)": f"{r:+.3f}"})
    res = pd.DataFrame(rows).sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    res.insert(0, "추천순위", range(1, len(res) + 1))
    render_table(res[["추천순위", "기간", "추천연도", "R2", "상관계수(r)"]],
                 headers=["추천순위", "기간", "추천연도", "R²", "상관계수(r)"], pct_cols=["R2"])

    # R² 추이 + 실제기온 기준선
    plot = res.sort_values("기간")
    ns = [int(s.replace("직전 ", "").replace("년", "")) for s in plot["기간"]]
    order = np.argsort(ns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[ns[i] for i in order], y=[plot["R2"].values[i] for i in order],
        mode="lines+markers+text", name="N년 평균기온 R²",
        text=[f"{plot['R2'].values[i]:.4f}" for i in order], textposition="top center",
        line=dict(color="#2563eb", width=2.5), marker=dict(size=9)))
    fig.add_hline(y=r2_actual, line=dict(color="#e8501a", dash="dash"),
                  annotation_text=f"당해 실제기온 R²={r2_actual:.4f}", annotation_position="bottom right")
    fig.update_layout(height=380, margin=dict(t=30, b=50, l=60, r=30),
                      xaxis=dict(title="평균에 사용한 직전 연수(N)", dtick=1),
                      yaxis=dict(title="R² (Poly-3)", tickformat=".3f"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False))

    # 상관도 매트릭스 (공급량 · 당해 실제기온 · N년 평균기온들)
    cols = ["공급량", "당해 실제기온"] + [f"{n}년평균" for n in range(1, nmax + 1)]
    corr = sample[cols].corr(method="pearson")
    corr.columns = corr.index = ["공급량", "당해 실제기온"] + [f"직전{n}년 평균기온" for n in range(1, nmax + 1)]
    st.markdown("**📐 상관도 매트릭스 (공급량 ↔ 기온 산출 방식)**")
    st.plotly_chart(_heatmap(corr, height=440), use_container_width=True,
                    config=dict(displaylogo=False), key=f"recent_temp_heatmap_{product}_{end_year}")

    best = res.iloc[0]
    diff = best["R2"] - r2_actual
    st.info(
        f"**해석** · 표본 {len(sample)}개월(공통 표본). R²가 가장 높은 방식은 **{best['추천연도']}** "
        f"(R²={best['R2']:.4f})이며, 당해 실제기온(R²={r2_actual:.4f})과의 차이는 {diff:+.4f}입니다. "
        "N년 평균기온은 그해 날씨의 편차를 평활화하므로 사후 적합도는 보통 실제기온보다 낮습니다. "
        "대신 미래 기온을 미리 알 수 없을 때 쓰는 '예상기온'의 대리값이므로, "
        "**R²가 실제기온 대비 크게 떨어지지 않는 N**이 예측용 기온 기준으로 적합합니다.")
    return res


# ══════════════════════════════════════════════
# B. 요인별 상관도 매트릭스 (기온 외 요인 포함)
# ══════════════════════════════════════════════
def render_factor_corr_matrix(merged, product):
    st.markdown("---")
    st.markdown('<div class="sub">🧩 요인별 상관도 매트릭스 (기온 외 요인 포함)</div>', unsafe_allow_html=True)
    st.caption(f"대상 상품: {product} 월별 공급량(MJ) × 요인 시트(기온·인구·세대·물가·청구전 등) · "
               "HDD18 = max(0, 18℃ − 평균기온) 파생값")

    factor_df, err = load_factor_sheet()
    if err or factor_df is None:
        st.warning(f"{err} — 구글시트가 '링크가 있는 모든 사용자 보기' 공유인지 확인해주세요.")
        return None

    base = merged[["연", "월", product]].rename(columns={product: "공급량"})
    df = base.merge(factor_df.drop(columns=["날짜", "총공급량"], errors="ignore"),
                    on=["연", "월"], how="inner")
    cols = [c for c in FACTOR_ORDER if c in df.columns]
    df = df.dropna(subset=["공급량"])
    if len(df) < 12 or len(cols) < 3:
        st.warning("요인 시트와 공급량이 겹치는 기간이 부족해 상관도를 계산할 수 없습니다.")
        return None
    st.caption(f"분석 기간 {int(df['연'].min())}-{int(df.loc[df['연'].idxmin(), '월']):02d} ~ "
               f"{int(df['연'].max())}-{int(df.loc[df['연'].idxmax(), '월']):02d} · 표본 {len(df)}개월")

    tab_p, tab_s = st.tabs(["Pearson (선형)", "Spearman (순위·비선형 단조)"])
    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")
    with tab_p:
        st.plotly_chart(_heatmap(corr_p, height=600), use_container_width=True,
                        config=dict(displaylogo=False), key=f"factor_heatmap_p_{product}")
    with tab_s:
        st.plotly_chart(_heatmap(corr_s, height=600), use_container_width=True,
                        config=dict(displaylogo=False), key=f"factor_heatmap_s_{product}")

    # 공급량과의 상관 순위
    rank = pd.DataFrame({"Pearson r": corr_p["공급량"], "Spearman ρ": corr_s["공급량"]}).drop(index="공급량")
    rank = rank.reindex(rank["Spearman ρ"].abs().sort_values(ascending=False).index)
    show = rank.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.3f}")
    show.insert(0, "요인", show.index)
    st.markdown("**공급량과의 상관 순위 (|Spearman ρ| 큰 순)**")
    st.markdown(show.to_html(index=False, classes="centered-table"), unsafe_allow_html=True)

    # 요인 간 다중공선성 경고
    others = [c for c in cols if c != "공급량"]
    pairs = []
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            v = corr_p.loc[a, b]
            if not np.isnan(v) and abs(v) >= 0.9:
                pairs.append((abs(v), a, b, v))
    pairs.sort(reverse=True)
    top = rank.index[:3].tolist()
    collinear = ", ".join(f"{a}↔{b}({v:+.2f})" for _, a, b, v in pairs[:5]) or "없음"
    st.info(
        f"**해석** · 공급량과 상관이 큰 요인은 **{', '.join(top)}** 순입니다. "
        "공급량–기온은 U자형(난방·냉방)이라 Pearson이 낮게 나올 수 있어 Spearman·HDD18을 함께 봅니다. "
        "총인구수·세대수·고령인구수·물가지수는 시간에 따라 완만히 변하는 추세 변수라, "
        "공급량이 계절로 크게 출렁이는 것과는 별개로 '연도 간 수준 변화'를 설명하는 요인입니다. "
        "즉 월별 상관이 낮아도 무의미한 것이 아니며, 반대로 추세끼리의 높은 상관은 우연일 수 있습니다. "
        f"요인 간 상관이 0.9 이상인 쌍: {collinear} — 다변량 회귀에 함께 넣으면 공선성에 주의하세요.")
    return corr_p
