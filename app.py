# app.py — 도시가스 공급량·판매량 예측
# Tab 1: 학습 기간 추천 (기온 학습 기간 + Poly-3 학습 기간)
# Tab 2: 공급량 예측 (Poly-3 + Normal/Best/Conservative)
# Tab 3: 냉난방공조용 예측 (GHP — 공급량/판매량 기반 + 기온방식 비교)
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="도시가스 공급량·판매량 예측", page_icon="📊", layout="wide")

# ══════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════
# ── 구글 스프레드시트 ID ──
SHEET1_ID = "1gIhArPlLBJ9fwlaqXtZWxiKlSK9hbRuz6HcDw_Yf7Is"   # 상품별 공급량 실적
SHEET2_ID = "13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs"   # 일별 기온/공급량
SHEET3_ID = "1-8RIPIkjnVXxoh5QJs6598nnHkWOGmrO655jr3b3g04"   # 상품별 판매량 실적

SHEET1_URL = f"https://docs.google.com/spreadsheets/d/{SHEET1_ID}/export?format=csv&gid=0"
SHEET2_URL = f"https://docs.google.com/spreadsheets/d/{SHEET2_ID}/export?format=csv&gid=0"
SHEET3_URL = f"https://docs.google.com/spreadsheets/d/{SHEET3_ID}/export?format=csv&gid=0"

# ── 상품 목록 (Sheet 1 기준) ──
HOUSING_PRODUCTS = ["취사용", "개별난방용", "중앙난방용", "자가열전용"]
OTHER_PRODUCTS   = ["일반용", "냉난방공조용", "업무난방용", "산업용",
                    "수송용", "열병합용", "연료전지용", "열전용설비용", "주한미군"]
PRODUCT_LIST     = HOUSING_PRODUCTS + OTHER_PRODUCTS
N_HOUSING = len(HOUSING_PRODUCTS)
N_OTHER   = len(OTHER_PRODUCTS)
BIO_SKIP_AFTER_N_OTHER = 5   # 수송용 다음에 BIO가스 행 건너뛰기
DATA_START_COL  = 3           # 상품별분배 데이터 시작열 (D열=3)

# ── 테이블 제목 검색 키워드 (Sheet 1) ──
SUPPLY_TITLE_VARIANTS = ["상품별 분배", "상품별분배"]

# ── 판매량 상품 매핑 (Sheet 3 → Sheet 1 이름) ──
SALES_TO_SUPPLY_NAME = {"냉방용": "냉난방공조용"}  # Sheet 3에서 '냉방용' = Sheet 1 '냉난방공조용'

MONTH_KR = [f"{m}월" for m in range(1, 13)]

# ══════════════════════════════════════════════
# 스타일
# ══════════════════════════════════════════════
st.markdown("""
<style>
h1 { color: #1a3c5e; border-bottom: 3px solid #e8501a; padding-bottom: 0.3rem; }
.sub { font-size:1.05rem; font-weight:600; color:#2c5f8a; margin:1rem 0 0.3rem 0; }
.info-box {
    background:#f4f8fc; border-radius:8px; padding:0.9rem 1.4rem;
    margin-bottom:1rem; border-left:4px solid #2c5f8a;
    font-size:0.92rem; line-height:1.8;
}
table.centered-table { width:100%; table-layout:fixed; }
table.centered-table th, table.centered-table td { text-align:center !important; }
/* 사이드바 메뉴 스타일 */
section[data-testid="stSidebar"] .stRadio > label { font-weight: 600; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── 공통 그래프 레이아웃 ──
CHART_LAYOUT = dict(
    font=dict(family="Pretendard, -apple-system, sans-serif", size=13),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, b=50, l=60, r=30),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.18,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=False, showline=True,
        linecolor="rgba(0,0,0,0.15)", linewidth=1,
        tickfont=dict(size=12),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=1,
        showline=False, zeroline=False,
        tickfont=dict(size=12),
    ),
)

# ══════════════════════════════════════════════
# 데이터 로드 함수
# ══════════════════════════════════════════════

def _find_row_containing(raw, text_variants, search_cols=(0, 1, 2, 3)):
    """raw(DataFrame, header=None)에서 텍스트 포함 행 찾기"""
    max_col = min(raw.shape[1], max(search_cols) + 1)
    for r in range(len(raw)):
        for c in range(max_col):
            val = raw.iat[r, c]
            if pd.isna(val):
                continue
            for t in text_variants:
                if t in str(val):
                    return r
    return None


def _extract_supply_table(raw):
    title_idx = _find_row_containing(raw, SUPPLY_TITLE_VARIANTS)
    if title_idx is None:
        return None, "상품별 분배 테이블을 찾을 수 없습니다."
    header_idx = title_idx + 1
    housing_rows = [header_idx + i for i in range(1, N_HOUSING + 1)]
    subtotal1_row = header_idx + N_HOUSING + 1
    other_before = [subtotal1_row + i for i in range(1, BIO_SKIP_AFTER_N_OTHER + 1)]
    bio_row = subtotal1_row + BIO_SKIP_AFTER_N_OTHER + 1
    n_after = N_OTHER - BIO_SKIP_AFTER_N_OTHER
    other_after = [bio_row + i for i in range(1, n_after + 1)]
    other_rows = other_before + other_after
    data_rows = housing_rows + other_rows
    dates = pd.to_datetime(raw.iloc[header_idx, DATA_START_COL:], errors="coerce")
    valid_cols = [i for i, d in enumerate(dates) if pd.notna(d)]
    dates_valid = dates.iloc[valid_cols]
    if len(valid_cols) == 0:
        return None, "헤더 행에서 날짜를 인식하지 못했습니다."
    result = {}
    for idx, row_i in enumerate(data_rows):
        if row_i >= len(raw):
            continue
        product = PRODUCT_LIST[idx]
        vals = pd.to_numeric(
            raw.iloc[row_i, DATA_START_COL:].iloc[valid_cols]
            .astype(str).str.replace(",", ""), errors="coerce"
        ).values
        result[product] = vals
    df = pd.DataFrame(result, index=dates_valid)
    df.index.name = "날짜"
    return df, None


@st.cache_data(ttl=1800)
def load_sheet1_supply():
    try:
        resp = requests.get(SHEET1_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text), header=None)
    except Exception as e:
        return None, f"Sheet 1 로드 실패: {e}"
    df, err = _extract_supply_table(raw)
    if err:
        return None, err
    return df, None


@st.cache_data(ttl=1800)
def load_sheet2_temperature():
    try:
        resp = requests.get(SHEET2_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return None, f"Sheet 2 로드 실패: {e}"
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = None
    for c in raw.columns:
        if c in ["일자", "날짜", "date", "Date"]:
            date_col = c; break
    if date_col is None:
        return None, "Sheet 2에서 '일자' 열을 찾을 수 없습니다."
    temp_col = None
    for c in raw.columns:
        if "평균기온" in c or "기온" in c:
            temp_col = c; break
    if temp_col is None:
        return None, "Sheet 2에서 '평균기온' 열을 찾을 수 없습니다."
    supply_col = None
    for c in raw.columns:
        if "공급량" in c and "MJ" in c:
            supply_col = c; break
    df = pd.DataFrame()
    df["일자"] = pd.to_datetime(raw[date_col], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        raw[temp_col].astype(str).str.replace(",", ""), errors="coerce")
    if supply_col:
        df["공급량_MJ"] = pd.to_numeric(
            raw[supply_col].astype(str).str.replace(",", ""), errors="coerce")
    for label in ["최저", "최고"]:
        for c in raw.columns:
            if label in c:
                df[label] = pd.to_numeric(
                    raw[c].astype(str).str.replace(",", ""), errors="coerce")
                break
    df = df.dropna(subset=["일자", "평균기온"]).sort_values("일자").reset_index(drop=True)
    df["연"] = df["일자"].dt.year
    df["월"] = df["일자"].dt.month
    df["일"] = df["일자"].dt.day
    return df, None


@st.cache_data(ttl=1800)
def load_sheet3_sales():
    try:
        resp = requests.get(SHEET3_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return None, f"Sheet 3 로드 실패: {e}"
    raw.columns = [str(c).strip() for c in raw.columns]
    if "연" not in raw.columns and "년" in raw.columns:
        raw.rename(columns={"년": "연"}, inplace=True)
    for c in raw.columns:
        if c not in ["일자", "날짜", "date"]:
            raw[c] = pd.to_numeric(
                raw[c].astype(str).str.replace(",", ""), errors="coerce")
    if "날짜" not in raw.columns and "일자" not in raw.columns:
        if "연" in raw.columns and "월" in raw.columns:
            raw["날짜"] = pd.to_datetime(
                raw["연"].astype(int).astype(str) + "-" +
                raw["월"].astype(int).astype(str) + "-01", errors="coerce")
    raw = raw.dropna(subset=["연", "월"]).reset_index(drop=True)
    raw["연"] = raw["연"].astype(int)
    raw["월"] = raw["월"].astype(int)
    return raw, None


def get_monthly_avg_temp(temp_daily: pd.DataFrame) -> pd.DataFrame:
    return (
        temp_daily.groupby(["연", "월"])["평균기온"]
        .mean().reset_index()
        .rename(columns={"평균기온": "월평균기온"})
    )


def get_cooling_period_temp(temp_daily: pd.DataFrame) -> pd.DataFrame:
    """
    검침 기간 평균기온: 전월 16일~말일 + 당월 1일~15일.
    반환: DataFrame(columns=[연, 월, 검침기온])
    """
    rows = []
    for (y, m), _ in temp_daily.groupby(["연", "월"]):
        cur_half = temp_daily[
            (temp_daily["연"] == y) & (temp_daily["월"] == m) & (temp_daily["일"] <= 15)
        ]["평균기온"]
        if m == 1:
            py, pm = y - 1, 12
        else:
            py, pm = y, m - 1
        prev_half = temp_daily[
            (temp_daily["연"] == py) & (temp_daily["월"] == pm) & (temp_daily["일"] >= 16)
        ]["평균기온"]
        combined = pd.concat([prev_half, cur_half])
        if len(combined) >= 5:
            rows.append({"연": int(y), "월": int(m), "검침기온": combined.mean()})
    return pd.DataFrame(rows)


def get_cooling_period_daily_temps(temp_daily: pd.DataFrame) -> dict:
    """
    검침기간별 일별 기온 목록 (일별합산 예측용).
    반환: {(연, 월): np.array([일별 평균기온])}
    """
    result = {}
    for (y, m), _ in temp_daily.groupby(["연", "월"]):
        cur_half = temp_daily[
            (temp_daily["연"] == y) & (temp_daily["월"] == m) & (temp_daily["일"] <= 15)
        ]["평균기온"].dropna().values
        if m == 1:
            py, pm = y - 1, 12
        else:
            py, pm = y, m - 1
        prev_half = temp_daily[
            (temp_daily["연"] == py) & (temp_daily["월"] == pm) & (temp_daily["일"] >= 16)
        ]["평균기온"].dropna().values
        combined = np.concatenate([prev_half, cur_half])
        if len(combined) >= 5:
            result[(int(y), int(m))] = combined
    return result


def merge_supply_and_temp(supply_df, temp_monthly):
    flat = supply_df.copy()
    flat["연"] = flat.index.year
    flat["월"] = flat.index.month
    flat = flat.reset_index(drop=True)
    merged = flat.merge(temp_monthly, on=["연", "월"], how="inner")
    product_cols = [c for c in merged.columns if c in PRODUCT_LIST]
    merged = merged[merged[product_cols].sum(axis=1) > 0].reset_index(drop=True)
    return merged


# ══════════════════════════════════════════════
# Poly-3 모델 함수
# ══════════════════════════════════════════════

def fit_poly3(x_train, y_train, x_pred):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_tr, y_tr = x_train[m], y_train[m]
    if len(x_tr) < 4:
        return np.full_like(x_pred, np.nan), 0.0, None, None
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_tr.reshape(-1, 1))
    model = LinearRegression().fit(Xtr, y_tr)
    r2 = model.score(Xtr, y_tr)
    y_pred = model.predict(poly.transform(x_pred.reshape(-1, 1)))
    return y_pred, r2, model, poly


def poly_eq_text(model):
    if model is None:
        return ""
    c = model.coef_
    c1, c2, c3 = (c[0] if len(c) > 0 else 0,
                   c[1] if len(c) > 1 else 0,
                   c[2] if len(c) > 2 else 0)
    d = model.intercept_
    return f"y = {c3:+,.4f}x³ {c2:+,.4f}x² {c1:+,.4f}x {d:+,.4f}"


def predict_daily_avg(model, poly, daily_temps):
    """
    일별예측합산 방식: 각 일별 기온으로 Poly-3 예측 후 평균.
    model/poly: 월 단위 데이터로 학습된 Poly-3 모델.
    daily_temps: 검침기간 내 일별 기온 배열.
    반환: 일별 예측값의 평균 (= 월 예측값).
    """
    if model is None or poly is None or daily_temps is None or len(daily_temps) == 0:
        return np.nan
    X = poly.transform(daily_temps.reshape(-1, 1))
    preds = model.predict(X)
    return float(np.mean(preds))


def recommend_train_ranges(merged_df, product, end_year=None):
    if end_year is None:
        end_year = int(merged_df["연"].max())
    min_year = int(merged_df["연"].min())
    actual = merged_df[merged_df["연"] == end_year][["월", "월평균기온", product]].dropna()
    if actual.empty or len(actual) < 3:
        return pd.DataFrame()
    x_actual = actual["월평균기온"].values.astype(float)
    y_actual = actual[product].values.astype(float)
    rows = []
    for sy in range(min_year, end_year):
        n_years = end_year - sy
        train = merged_df[
            (merged_df["연"] >= sy) & (merged_df["연"] < end_year)
        ][["월평균기온", product]].dropna()
        if len(train) < 12:
            rows.append({"시작연도": sy, "종료연도": end_year - 1,
                "기간": f"{sy}~{end_year - 1}", "추천연도": f"최근 {n_years}년", "R2": np.nan})
            continue
        x_tr = train["월평균기온"].values.astype(float)
        y_tr = train[product].values.astype(float)
        y_pred, _, _, _ = fit_poly3(x_tr, y_tr, x_actual)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"시작연도": sy, "종료연도": end_year - 1,
            "기간": f"{sy}~{end_year - 1}", "추천연도": f"최근 {n_years}년",
            "R2": float(r2) if not np.isnan(r2) else np.nan})
    df = pd.DataFrame(rows)
    df = df.sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    return df


def recommend_temp_period(merged_df, product, temp_daily, end_year=None,
                          fixed_train_years=3):
    if end_year is None:
        end_year = int(merged_df["연"].max())
    train_start = end_year - fixed_train_years
    train = merged_df[
        (merged_df["연"] >= train_start) & (merged_df["연"] < end_year)
    ][["월평균기온", product]].dropna()
    if len(train) < 12:
        return pd.DataFrame()
    x_tr = train["월평균기온"].values.astype(float)
    y_tr = train[product].values.astype(float)
    actual = merged_df[merged_df["연"] == end_year][["월", product]].dropna()
    if actual.empty or len(actual) < 3:
        return pd.DataFrame()
    y_actual = actual[product].values.astype(float)
    temp_by_ym = temp_daily.groupby(["연", "월"])["평균기온"].mean().reset_index()
    periods = [1, 2, 3, 5]
    results = []
    for n_years in periods:
        past_years = list(range(end_year - n_years, end_year))
        period_str = f"{min(past_years)}~{max(past_years)}" if len(past_years) > 1 else str(past_years[0])
        rec_label = f"과거 {n_years}년평균"
        past_temps = temp_by_ym[temp_by_ym["연"].isin(past_years)]
        if past_temps.empty:
            results.append({"기간": period_str, "추천연도": rec_label, "R2": np.nan}); continue
        avg_by_month = past_temps.groupby("월")["평균기온"].mean().reset_index()
        avg_by_month.rename(columns={"평균기온": "예측기온"}, inplace=True)
        compare = actual[["월"]].merge(avg_by_month, on="월", how="inner")
        if len(compare) < 3 or len(compare) != len(y_actual):
            results.append({"기간": period_str, "추천연도": rec_label, "R2": np.nan}); continue
        x_pred = compare["예측기온"].values.astype(float)
        y_pred, _, _, _ = fit_poly3(x_tr, y_tr, x_pred)
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        results.append({"기간": period_str, "추천연도": rec_label, "R2": float(r2)})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════

def render_centered_table(df, float_cols=None, int_cols=None, pct_cols=None, index=False):
    float_cols = float_cols or []; int_cols = int_cols or []; pct_cols = pct_cols or []
    show = df.copy()
    for c in float_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.2f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{int(round(x)):,}")
    for c in pct_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}")
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)


def _render_highlight_table(df, headers=None, pct_cols=None):
    pct_cols = pct_cols or []
    cols = list(df.columns)
    if headers is None:
        headers = cols
    html_rows = ""
    for _, row in df.iterrows():
        rank = row.iloc[0]
        if rank == 1:
            style = ' style="background-color:#e8f4fd; font-weight:bold;"'
        else:
            style = ""
        html_rows += f"<tr{style}>"
        for c in cols:
            v = row[c]
            if c in pct_cols:
                v = "" if pd.isna(v) else f"{float(v):.4f}"
            html_rows += f"<td>{v}</td>"
        html_rows += "</tr>"
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    st.markdown(f"""
    <table class="centered-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{html_rows}</tbody>
    </table>""", unsafe_allow_html=True)


def _make_scatter_chart(x_train, y_train, title, xlab, ylab, r2):
    """공통 산점도 + Poly-3 회귀곡선 생성"""
    _v = (~np.isnan(x_train)) & (~np.isnan(y_train))
    xv, yv = x_train[_v], y_train[_v]
    xx = np.linspace(xv.min() - 2, xv.max() + 2, 200)
    yy, _, _, _ = fit_poly3(xv, yv, xx)
    pred_tr, _, _, _ = fit_poly3(xv, yv, xv)
    resid_std = np.std(yv - pred_tr) if len(yv) > 1 else 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", name="학습 샘플",
        marker=dict(size=6, opacity=0.6, color="#3b82f6"),
        hovertemplate=f"{xlab}=%{{x:.1f}}<br>{ylab}=%{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Poly-3 회귀",
        line=dict(color="#e8501a", width=2.5)))
    if resid_std > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([xx, xx[::-1]]),
            y=np.concatenate([yy + 1.96 * resid_std, (yy - 1.96 * resid_std)[::-1]]),
            fill="toself", fillcolor="rgba(232,80,26,0.10)",
            line=dict(width=0), name="95% 신뢰구간"))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(text=f"{title} (R²={r2:.4f})", font=dict(size=14, color="#1f2937")),
        xaxis_title=xlab, yaxis_title=ylab,
        margin=dict(t=50, b=70, l=60, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.13,
                    xanchor="center", x=0.5, font=dict(size=10)),
        height=420)
    return fig


def _make_line_chart(traces_data, title, xlab, ylab, height=420):
    """
    공통 월별 라인 차트.
    traces_data: list of dict(x, y, name, color, dash, customdata, hover_extra)
    """
    fig = go.Figure()
    for t in traces_data:
        line_kw = dict(color=t.get("color", "#2563eb"), width=t.get("width", 2.5))
        if t.get("dash"):
            line_kw["dash"] = t["dash"]
        scatter_kw = dict(
            x=t["x"], y=t["y"], name=t["name"],
            mode="lines+markers", line=line_kw,
            marker=dict(size=t.get("marker_size", 7)),
        )
        if t.get("customdata") is not None:
            scatter_kw["customdata"] = t["customdata"]
            extra = t.get("hover_extra", "")
            scatter_kw["hovertemplate"] = f"%{{x}} %{{y:,.0f}}{extra}<extra></extra>"
        else:
            scatter_kw["hovertemplate"] = f"%{{x}} %{{y:,.0f}}<extra></extra>"
        fig.add_trace(go.Scatter(**scatter_kw))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1f2937")),
        xaxis_title=xlab, yaxis_title=ylab,
        yaxis_rangemode="tozero",
        margin=dict(t=50, b=70, l=60, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.13,
                    xanchor="center", x=0.5, font=dict(size=10)),
        height=height, dragmode="pan")
    return fig


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════

def main():
    st.title("📊 도시가스 공급량·판매량 예측")
    st.caption("대성에너지(주) 마케팅본부 · Poly-3 기온↔공급량/판매량 회귀 모델")

    # ── 데이터 로드 ──
    supply_df, err1 = load_sheet1_supply()
    temp_daily, err2 = load_sheet2_temperature()
    sales_df, err3   = load_sheet3_sales()

    if err1 or err2:
        st.error("공급량 또는 기온 데이터를 불러오지 못했습니다. 구글시트 공유 설정을 확인해주세요.")
        st.stop()

    temp_monthly = get_monthly_avg_temp(temp_daily)
    merged = merge_supply_and_temp(supply_df, temp_monthly)

    if merged.empty:
        st.warning("공급량과 기온 데이터의 겹치는 기간이 없습니다.")
        st.stop()

    available_products = [p for p in PRODUCT_LIST if p in merged.columns]
    years_all = sorted(merged["연"].unique().astype(int))

    # ── 사이드바 ──
    with st.sidebar:
        st.markdown("### 📋 메뉴")
        menu_options = [
            "🎯 학습 데이터 기간 추천",
            "📈 공급량 예측",
            "🧊 판매량 예측 (냉방용)",
        ]
        selected_menu = st.radio(
            "분석 메뉴", options=menu_options,
            index=0, label_visibility="collapsed", key="main_menu")

        st.markdown("---")
        st.markdown("### 🌡️ 예상기온 업로드")
        st.caption("미래 기온 예측값 (엑셀: 날짜/연·월, 예상기온 열)")
        uploaded_temp = st.file_uploader(
            "예상기온 엑셀 (.xlsx)", type=["xlsx", "xls", "csv"],
            key="upload_forecast_temp")
        forecast_temp_df = None
        if uploaded_temp is not None:
            forecast_temp_df = _parse_uploaded_temp(uploaded_temp)
            if forecast_temp_df is not None:
                st.success(f"✅ 예상기온 {len(forecast_temp_df)}개월 로드")

        st.markdown("---")
        with st.expander("📥 데이터 로드 상태", expanded=False):
            if err1:
                st.error(f"❌ 공급량: {err1}")
            else:
                st.success(f"✅ 공급량 ({len(supply_df.columns)}개 상품, {len(supply_df)}개월)")
            if err2:
                st.error(f"❌ 기온: {err2}")
            else:
                st.success(f"✅ 일별기온 ({len(temp_daily):,}일)")
            if err3:
                st.error(f"❌ 판매량: {err3}")
            else:
                st.success(f"✅ 판매량 ({len(sales_df)}개월)")
            st.caption(f"데이터 기간: {min(years_all)}~{max(years_all)}년 · 월 데이터 {len(merged)}건")

    # ══════════════════════════════════════════
    # ── TAB 1: 학습 기간 추천 ──
    # ══════════════════════════════════════════
    if selected_menu == menu_options[0]:
        st.markdown("### 🎯 학습 기간 추천")
        st.markdown("""
        <div class="info-box">
        <b>Poly-3 학습 기간</b>: 기온 고정(실적연도 실제 기온) · Poly-3 학습 기간만 변경 → 예측 vs 실적 R²<br>
        <b>기온 학습 기간</b>: Poly-3 고정(최근 3년 학습) · 기온만 과거 N년 평균으로 변경 → 예측 vs 실적 R²
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            rec_product = st.selectbox("대상 상품", options=available_products,
                index=available_products.index("개별난방용") if "개별난방용" in available_products else 0,
                key="rec_product")
        with c2:
            rec_end_year = st.selectbox("실적연도", options=years_all,
                index=len(years_all) - 1, key="rec_end_year")

        if st.button("🔎 추천 구간 계산", type="primary", key="btn_rec"):
            st.markdown('<div class="sub">📊 Poly-3 학습 기간 추천</div>', unsafe_allow_html=True)
            st.caption(f"기온 고정: {rec_end_year}년 실제 월별 기온 사용 · Poly-3 학습 기간만 변경")
            rec_df = recommend_train_ranges(merged, rec_product, end_year=rec_end_year)
            rec_all = rec_df.copy()
            rec_all.insert(0, "추천순위", range(1, len(rec_all) + 1))
            _render_highlight_table(
                rec_all[["추천순위", "기간", "추천연도", "R2"]],
                headers=["추천순위", "기간", "추천연도", "R²"], pct_cols=["R2"])

            # ── R² 그래프 ──
            rec_plot = rec_df.sort_values("시작연도")
            fig_r = go.Figure()
            top3 = rec_all.head(3)
            highlight_colors = [
                ("rgba(59,130,246,0.12)", "rgba(59,130,246,0.5)"),
                ("rgba(16,185,129,0.10)", "rgba(16,185,129,0.45)"),
                ("rgba(139,92,246,0.08)", "rgba(139,92,246,0.4)")]
            rank_labels = ["1위 추천", "2위 추천", "3위 추천"]
            for i, (_, row) in enumerate(top3.iterrows()):
                if i >= 3: break
                fill_c, border_c = highlight_colors[i]
                sy, ey = int(row["시작연도"]), int(row["종료연도"])
                fig_r.add_shape(type="rect", xref="x", yref="paper",
                    x0=sy-0.4, x1=ey+0.4, y0=0, y1=1,
                    line=dict(width=1.5, color=border_c, dash="dot"),
                    fillcolor=fill_c, layer="below")
                fig_r.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=fill_c,
                                line=dict(width=1.5, color=border_c), symbol="square"),
                    name=f"{rank_labels[i]} ({row['추천연도']})", showlegend=True))

            r2_vals = rec_plot["R2"].dropna().values
            data_min, data_max = float(r2_vals.min()), float(r2_vals.max())
            data_range = max(data_max - data_min, 0.0005)
            chart_range = data_range / 0.70
            padding = (chart_range - data_range) / 2
            y_min = max(0, data_min - padding)
            y_max = min(1.0, data_max + padding)

            best_idx = rec_plot["R2"].idxmax()
            best_x, best_y = rec_plot.loc[best_idx, "시작연도"], rec_plot.loc[best_idx, "R2"]

            fig_r.add_trace(go.Scatter(
                x=rec_plot["시작연도"], y=rec_plot["R2"],
                mode="lines+markers+text",
                text=[f"{v:.4f}" if pd.notna(v) else "" for v in rec_plot["R2"]],
                textposition="top center", textfont=dict(size=11, color="#374151"),
                name="R² (Poly-3)",
                hovertemplate="시작연도=%{x}<br>R²=%{y:.6f}<extra></extra>",
                line=dict(color="#2563eb", width=2.5, shape="spline"),
                marker=dict(size=9, color="#2563eb", line=dict(width=2, color="white"))))
            fig_r.add_trace(go.Scatter(
                x=rec_plot["시작연도"], y=rec_plot["R2"],
                mode="lines", showlegend=False, line=dict(width=0),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"))
            fig_r.add_trace(go.Scatter(
                x=[best_x], y=[best_y], mode="markers",
                marker=dict(size=14, color="#f59e0b", line=dict(width=2.5, color="white"), symbol="star"),
                name=f"최적 (R²={best_y:.4f})",
                hovertemplate=f"최적 시작연도={best_x}<br>R²={best_y:.6f}<extra></extra>"))
            fig_r.update_layout(**CHART_LAYOUT)
            fig_r.update_layout(
                title=dict(text=f"학습 시작연도별 R² — {rec_product} (실적연도={rec_end_year})",
                           font=dict(size=15, color="#1f2937")),
                xaxis_title="학습 시작연도", yaxis_title="R² (예측 vs 실적)",
                xaxis_tickmode="linear", xaxis_dtick=1,
                yaxis_range=[y_min, y_max], yaxis_tickformat=".4f",
                margin=dict(t=60, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.08)", borderwidth=1, font=dict(size=10)),
                height=480)
            st.plotly_chart(fig_r, use_container_width=True,
                            config=dict(scrollZoom=True, displaylogo=False))

            # ── 기온 학습 기간 추천 ──
            st.markdown('<div class="sub">🌡️ 기온 학습 기간 추천</div>', unsafe_allow_html=True)
            st.caption(f"Poly-3 고정: 최근 3년 학습 · 기온만 과거 N년 월별 평균으로 변경하여 {rec_end_year}년 실적과 비교")
            temp_rec = recommend_temp_period(merged, rec_product, temp_daily, end_year=rec_end_year)
            if not temp_rec.empty:
                temp_rec_show = temp_rec[["기간", "추천연도", "R2"]].copy()
                temp_rec_show = temp_rec_show.sort_values("R2", ascending=False, na_position="last")
                temp_rec_show.insert(0, "추천순위", range(1, len(temp_rec_show) + 1))
                _render_highlight_table(temp_rec_show,
                    headers=["추천순위", "기간", "추천연도", "R²"], pct_cols=["R2"])
                best_temp = temp_rec_show.iloc[0]
                st.info(f"🏆 **추천 기온 기간**: {best_temp['추천연도']} ({best_temp['기간']}, R²={best_temp['R2']:.4f})")
            else:
                st.warning("기온 학습 기간 추천 계산에 필요한 데이터가 부족합니다.")

            if not rec_df.empty:
                best_poly = rec_df.iloc[0]
                st.success(f"📌 **종합 추천**: Poly-3 학습 기간 **{best_poly['기간']}** "
                           f"({best_poly['추천연도']}, R²={best_poly['R2']:.4f})")

    # ══════════════════════════════════════════
    # ── TAB 2: 공급량 예측 ──
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[1]:
        st.markdown("### 📈 공급량 예측 (Poly-3)")
        c1, c2 = st.columns(2)
        with c1:
            pred_products = st.multiselect("예측 상품 선택", options=available_products,
                default=["개별난방용"] if "개별난방용" in available_products else available_products[:1],
                key="pred_products")
        with c2:
            train_years = st.multiselect("학습 연도 선택", options=years_all,
                default=years_all[-3:] if len(years_all) >= 3 else years_all,
                key="train_years")

        st.markdown('<div class="sub">🌡️ 시나리오 Δ°C (예상기온 보정)</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.1, format="%.1f", key="d_norm")
        with sc2:
            d_best = st.number_input("Best Δ°C", value=-1.0, step=0.1, format="%.1f", key="d_best")
        with sc3:
            d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.1, format="%.1f", key="d_cons")

        st.markdown('<div class="sub">📅 예측 기간</div>', unsafe_allow_html=True)
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            pred_start_y = st.selectbox("시작 연도", list(range(2020, 2036)), index=6, key="pred_sy")
        with pc2:
            pred_start_m = st.selectbox("시작 월", list(range(1, 13)), index=0, key="pred_sm")
        with pc3:
            pred_end_y = st.selectbox("종료 연도", list(range(2020, 2036)), index=6, key="pred_ey")
        with pc4:
            pred_end_m = st.selectbox("종료 월", list(range(1, 13)), index=11, key="pred_em")

        if st.button("🧮 공급량 예측 실행", type="primary", key="btn_supply_pred"):
            if not pred_products:
                st.warning("예측할 상품을 선택해주세요."); st.stop()
            if not train_years:
                st.warning("학습 연도를 선택해주세요."); st.stop()
            train_data = merged[merged["연"].isin(train_years)]
            if len(train_data) < 12:
                st.error("학습 데이터가 12건 미만입니다. 학습 연도를 추가해주세요."); st.stop()
            x_train = train_data["월평균기온"].values.astype(float)
            f_start = pd.Timestamp(year=pred_start_y, month=pred_start_m, day=1)
            f_end   = pd.Timestamp(year=pred_end_y,   month=pred_end_m,   day=1)
            if f_end < f_start:
                st.error("예측 종료가 시작보다 앞입니다."); st.stop()
            fut_months = pd.date_range(start=f_start, end=f_end, freq="MS")
            fut_df = pd.DataFrame({"연": fut_months.year, "월": fut_months.month})
            monthly_avg = train_data.groupby("월")["월평균기온"].mean()
            if forecast_temp_df is not None:
                fut_df = fut_df.merge(forecast_temp_df[["연", "월", "예상기온"]],
                    on=["연", "월"], how="left")
                miss = fut_df["예상기온"].isna()
                if miss.any():
                    fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(monthly_avg)
            else:
                fut_df["예상기온"] = fut_df["월"].map(monthly_avg)
            if fut_df["예상기온"].isna().any():
                st.warning("일부 월의 예상기온을 결정하지 못했습니다.")
                overall_avg = train_data.groupby("월")["월평균기온"].mean()
                miss = fut_df["예상기온"].isna()
                fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(overall_avg)
            scenarios = {"Normal": d_norm, "Best": d_best, "Conservative": d_cons}
            for prod in pred_products:
                y_train = train_data[prod].values.astype(float)
                st.markdown(f'<div class="sub">📦 {prod}</div>', unsafe_allow_html=True)
                _, r2_train, model, poly = fit_poly3(x_train, y_train, x_train)
                st.caption(f"Poly-3 Train R² = {r2_train:.4f} | {poly_eq_text(model)}")
                scenario_tables = {}
                for sname, delta in scenarios.items():
                    x_fut = (fut_df["예상기온"] + delta).values.astype(float)
                    y_pred, _, _, _ = fit_poly3(x_train, y_train, x_fut)
                    y_pred = np.clip(np.rint(y_pred).astype(np.int64), 0, None)
                    tbl = fut_df[["연", "월"]].copy()
                    tbl["예상기온"] = fut_df["예상기온"] + delta
                    tbl[prod] = y_pred
                    scenario_tables[sname] = tbl
                fig = go.Figure()
                for y in sorted(years_all)[-3:]:
                    act = merged[merged["연"] == y][["월", prod, "월평균기온"]].sort_values("월")
                    if act.empty: continue
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in act["월"]], y=act[prod],
                        customdata=np.round(act["월평균기온"].values, 2),
                        mode="lines+markers", name=f"{y} 실적",
                        hovertemplate="%{x} %{y:,.0f} MJ<br>기온 %{customdata:.1f}℃<extra></extra>"))
                for y in sorted(fut_df["연"].unique()):
                    tbl = scenario_tables["Normal"]
                    row = tbl[tbl["연"] == y].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in row["월"]], y=row[prod],
                        customdata=np.round(row["예상기온"].values, 2),
                        mode="lines", name=f"예측(Normal) {y}", line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,.0f} MJ<br>기온 %{customdata:.1f}℃<extra></extra>"))
                fig.update_layout(**CHART_LAYOUT)
                fig.update_layout(
                    title=f"{prod} — Poly-3 예측 (Train R²={r2_train:.4f})",
                    xaxis_title="월", yaxis_title="공급량 (MJ)", yaxis_rangemode="tozero",
                    margin=dict(t=60, b=80), dragmode="pan",
                    legend=dict(orientation="h", yanchor="top", y=-0.13,
                                xanchor="center", x=0.5, font=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True,
                                config=dict(scrollZoom=True, displaylogo=False))
                st.markdown(f'<div class="sub">📋 {prod} — 시나리오별 월별 예측</div>',
                            unsafe_allow_html=True)
                compare_tbl = fut_df[["연", "월"]].copy()
                for sname in scenarios:
                    compare_tbl[sname] = scenario_tables[sname][prod].values
                sum_row = {"연": "합계", "월": ""}
                for sname in scenarios:
                    sum_row[sname] = compare_tbl[sname].sum()
                compare_full = pd.concat([compare_tbl, pd.DataFrame([sum_row])], ignore_index=True)
                render_centered_table(compare_full, int_cols=list(scenarios.keys()))
                with st.expander(f"🔎 {prod} — 기온↔공급량 산점도 (학습 데이터)"):
                    fig_sc = _make_scatter_chart(x_train, y_train,
                        f"{prod} — 기온 vs 공급량", "기온 (℃)", "공급량 (MJ)", r2_train)
                    st.plotly_chart(fig_sc, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for sname in scenarios:
                    all_prods = fut_df[["연", "월"]].copy()
                    for prod in pred_products:
                        all_prods[prod] = scenario_tables[sname][prod].values if sname in scenario_tables else 0
                    all_prods.to_excel(writer, sheet_name=sname, index=False)
            st.download_button("⬇️ 예측 결과 엑셀 다운로드", data=buf.getvalue(),
                file_name="공급량_예측_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ══════════════════════════════════════════
    # ── TAB 3: 냉난방공조용 예측 (GHP) ──
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[2]:
        st.markdown("### 🧊 냉난방공조용 예측 (GHP)")
        st.markdown("""
        <div class="info-box">
        <b>대상</b>: 냉난방공조용 — GHP(가스히트펌프) 냉·난방 겸용 건물. 기온이 높아도 냉방 가동으로 사용량 발생.<br>
        <b>검침 기준 기온</b>: 전월 16일~말일 + 당월 1일~15일 평균기온<br>
        <b>기온 방식 비교</b>: ① 기간평균 방식 (기간 내 기온 평균 → 예측) vs ② 일별합산 방식 (일별 기온 → 일별 예측 → 평균)
        </div>
        """, unsafe_allow_html=True)

        if err3:
            st.error("판매량 데이터(Sheet 3)를 불러오지 못했습니다."); st.stop()

        # ── 데이터 준비 ──
        # 판매량: Sheet 3 냉방용
        cooling_col = None
        for c in sales_df.columns:
            if "냉방" in str(c) or "냉난방" in str(c):
                cooling_col = c; break
        if cooling_col is None:
            st.error("Sheet 3에서 '냉방용' 열을 찾을 수 없습니다."); st.stop()

        # 공급량: Sheet 1 냉난방공조용
        supply_product = "냉난방공조용"
        has_supply = (supply_df is not None and supply_product in supply_df.columns)

        # 검침기온
        cooling_temp = get_cooling_period_temp(temp_daily)
        daily_temps_dict = get_cooling_period_daily_temps(temp_daily)
        if cooling_temp.empty:
            st.error("검침기간 기온 계산 불가."); st.stop()

        # 판매량 + 검침기온 병합
        sales_clean = sales_df[["연", "월", cooling_col]].copy()
        sales_clean[cooling_col] = pd.to_numeric(sales_clean[cooling_col], errors="coerce")
        sales_merged = sales_clean.merge(cooling_temp, on=["연", "월"], how="inner")
        sales_merged = sales_merged.dropna(subset=[cooling_col, "검침기온"])
        sales_merged = sales_merged[sales_merged[cooling_col] > 0].reset_index(drop=True)

        # 공급량 + 검침기온 병합
        supply_merged = pd.DataFrame()
        if has_supply:
            sup_flat = supply_df[[supply_product]].copy()
            sup_flat["연"] = supply_df.index.year
            sup_flat["월"] = supply_df.index.month
            sup_flat = sup_flat.reset_index(drop=True)
            supply_merged = sup_flat.merge(cooling_temp, on=["연", "월"], how="inner")
            supply_merged = supply_merged.dropna(subset=[supply_product, "검침기온"])
            supply_merged = supply_merged[supply_merged[supply_product] > 0].reset_index(drop=True)

        sales_years = sorted(sales_merged["연"].unique().astype(int)) if not sales_merged.empty else []
        supply_years = sorted(supply_merged["연"].unique().astype(int)) if not supply_merged.empty else []
        all_cooling_years = sorted(set(sales_years) | set(supply_years))

        if not all_cooling_years:
            st.warning("데이터가 없습니다."); st.stop()

        info_parts = []
        if sales_years:
            info_parts.append(f"판매량 {min(sales_years)}~{max(sales_years)}년 ({len(sales_merged)}건)")
        if supply_years:
            info_parts.append(f"공급량 {min(supply_years)}~{max(supply_years)}년 ({len(supply_merged)}건)")
        st.caption(f"📦 사용 열: 판매량=**{cooling_col}** / 공급량=**{supply_product}** · " + " · ".join(info_parts))

        # ── 공통 설정 ──
        sc1, sc2 = st.columns(2)
        with sc1:
            cooling_train_years = st.multiselect(
                "학습 연도 선택", options=all_cooling_years, default=all_cooling_years,
                key="cooling_train_years")
        with sc2:
            cooling_pred_year = st.selectbox(
                "예측/비교 연도", options=list(range(min(all_cooling_years), 2036)),
                index=len(all_cooling_years) - 1, key="cooling_pred_year")

        st.markdown('<div class="sub">🌡️ 예측 기온 입력</div>', unsafe_allow_html=True)
        temp_input_mode = st.radio("방식 선택",
            ["학습기간 월평균 사용", "업로드한 예상기온 사용"],
            index=0, horizontal=True, key="cooling_temp_mode")

        if st.button("🧮 예측 실행", type="primary", key="btn_cooling_pred"):

            # ═══════════════════════════════════
            # 모델 학습
            # ═══════════════════════════════════
            # 판매량 모델
            train_sales = sales_merged[sales_merged["연"].isin(cooling_train_years)]
            has_sales_model = len(train_sales) >= 6
            if has_sales_model:
                x_s_tr = train_sales["검침기온"].values.astype(float)
                y_s_tr = train_sales[cooling_col].values.astype(float)
                _, r2_sales, model_sales, poly_sales = fit_poly3(x_s_tr, y_s_tr, x_s_tr)
            else:
                r2_sales, model_sales, poly_sales = 0, None, None

            # 공급량 모델
            train_supply = supply_merged[supply_merged["연"].isin(cooling_train_years)] if not supply_merged.empty else pd.DataFrame()
            has_supply_model = len(train_supply) >= 6
            if has_supply_model:
                x_sup_tr = train_supply["검침기온"].values.astype(float)
                y_sup_tr = train_supply[supply_product].values.astype(float)
                _, r2_supply, model_supply, poly_supply = fit_poly3(x_sup_tr, y_sup_tr, x_sup_tr)
            else:
                r2_supply, model_supply, poly_supply = 0, None, None

            # 예측 기온 결정
            pred_months = list(range(1, 13))
            if has_sales_model:
                monthly_avg_cool = train_sales.groupby("월")["검침기온"].mean()
            elif has_supply_model:
                monthly_avg_cool = train_supply.groupby("월")["검침기온"].mean()
            else:
                st.error("학습 데이터가 부족합니다."); st.stop()

            if temp_input_mode == "업로드한 예상기온 사용" and forecast_temp_df is not None:
                fc_yr = forecast_temp_df[forecast_temp_df["연"] == cooling_pred_year]
                pred_temps = []
                for m in pred_months:
                    row = fc_yr[fc_yr["월"] == m]
                    pred_temps.append(float(row.iloc[0]["예상기온"]) if not row.empty
                                      else monthly_avg_cool.get(m, np.nan))
            else:
                pred_temps = [monthly_avg_cool.get(m, np.nan) for m in pred_months]

            x_pred = np.array(pred_temps, dtype=float)
            valid_mask = ~np.isnan(x_pred)
            x_valid = x_pred[valid_mask]
            valid_months = [m for m, v in zip(pred_months, valid_mask) if v]

            if valid_mask.sum() == 0:
                st.error("예측 기온이 모두 비어있습니다."); st.stop()

            # Method A: 기간평균 예측
            pred_sales_A = pred_supply_A = None
            if has_sales_model:
                yp, _, _, _ = fit_poly3(x_s_tr, y_s_tr, x_valid)
                pred_sales_A = np.clip(np.rint(yp).astype(np.int64), 0, None)
            if has_supply_model:
                yp, _, _, _ = fit_poly3(x_sup_tr, y_sup_tr, x_valid)
                pred_supply_A = np.clip(np.rint(yp).astype(np.int64), 0, None)

            # Method B: 일별합산 예측
            pred_sales_B = []
            pred_supply_B = []
            for m in valid_months:
                dtk = daily_temps_dict.get((cooling_pred_year, m))
                if dtk is None:
                    # 과거 학습기간에서 해당 월 일별기온 평균 사용
                    fallback_temps = []
                    for yr in cooling_train_years:
                        dt = daily_temps_dict.get((yr, m))
                        if dt is not None:
                            fallback_temps.append(dt)
                    if fallback_temps:
                        max_len = max(len(t) for t in fallback_temps)
                        padded = [np.pad(t, (0, max_len - len(t)), constant_values=np.nan) for t in fallback_temps]
                        dtk = np.nanmean(np.array(padded), axis=0)
                        dtk = dtk[~np.isnan(dtk)]
                if dtk is not None and len(dtk) > 0:
                    if has_sales_model:
                        pred_sales_B.append(predict_daily_avg(model_sales, poly_sales, dtk))
                    else:
                        pred_sales_B.append(np.nan)
                    if has_supply_model:
                        pred_supply_B.append(predict_daily_avg(model_supply, poly_supply, dtk))
                    else:
                        pred_supply_B.append(np.nan)
                else:
                    pred_sales_B.append(np.nan)
                    pred_supply_B.append(np.nan)
            pred_sales_B = np.array(pred_sales_B)
            pred_supply_B = np.array(pred_supply_B)

            # 실적 데이터 (비교용)
            actual_sales_yr = sales_merged[sales_merged["연"] == cooling_pred_year]
            actual_supply_yr = supply_merged[supply_merged["연"] == cooling_pred_year] if not supply_merged.empty else pd.DataFrame()

            # ═══════════════════════════════════
            # 서브탭 표시
            # ═══════════════════════════════════
            sub1, sub2, sub3, sub4 = st.tabs([
                "1️⃣ 공급량 기반 예측",
                "2️⃣ 판매량 기반 예측",
                "3️⃣ 실적 vs 예측 비교",
                "4️⃣ 기온방식 비교 (평균 vs 일별)",
            ])

            # ─────────────────────────────────
            # SUB 1: 공급량 기반 예측
            # ─────────────────────────────────
            with sub1:
                if not has_supply_model:
                    st.warning("공급량 학습 데이터가 부족하여 예측할 수 없습니다.")
                else:
                    st.markdown(f'<div class="sub">📦 {supply_product} — 공급량 기반 Poly-3</div>',
                                unsafe_allow_html=True)
                    st.caption(f"Train R² = {r2_supply:.4f} | {poly_eq_text(model_supply)}")

                    # 결과 테이블
                    tbl_sup = pd.DataFrame({"월": [f"{m}월" for m in pred_months]})
                    tbl_sup["검침기온"] = pred_temps
                    tbl_sup["예측_공급량"] = np.nan
                    j = 0
                    for i, v in enumerate(valid_mask):
                        if v:
                            tbl_sup.loc[i, "예측_공급량"] = int(pred_supply_A[j])
                            j += 1

                    if not actual_supply_yr.empty:
                        act_s = actual_supply_yr.set_index("월")[supply_product]
                        tbl_sup["실적_공급량"] = [act_s.get(m, np.nan) for m in pred_months]
                        tbl_sup["차이"] = pd.to_numeric(tbl_sup["예측_공급량"], errors="coerce") - pd.to_numeric(tbl_sup["실적_공급량"], errors="coerce")

                    sum_row = {"월": "합계", "검침기온": ""}
                    for c in tbl_sup.columns:
                        if c not in ["월", "검침기온"]:
                            sum_row[c] = pd.to_numeric(tbl_sup[c], errors="coerce").sum()
                    tbl_sup_full = pd.concat([tbl_sup, pd.DataFrame([sum_row])], ignore_index=True)
                    int_c = [c for c in tbl_sup_full.columns if c not in ["월", "검침기온"]]
                    render_centered_table(tbl_sup_full, float_cols=["검침기온"], int_cols=int_c)

                    # 라인 차트
                    traces = []
                    for y in sorted(supply_years)[-3:]:
                        act = train_supply[train_supply["연"] == y].sort_values("월")
                        if not act.empty:
                            traces.append(dict(
                                x=[f"{int(m)}월" for m in act["월"]], y=act[supply_product].values,
                                name=f"{y} 실적", color=None, customdata=np.round(act["검침기온"].values, 1),
                                hover_extra="<br>검침기온 %{customdata:.1f}℃"))
                    traces.append(dict(
                        x=[f"{m}월" for m in valid_months], y=pred_supply_A,
                        name=f"예측 {cooling_pred_year}", color="#e8501a", dash="dash",
                        customdata=np.round(x_valid, 1),
                        hover_extra="<br>검침기온 %{customdata:.1f}℃"))
                    colors_cycle = ["#2563eb", "#06b6d4", "#f59e0b", "#8b5cf6"]
                    for i, t in enumerate(traces):
                        if t.get("color") is None:
                            t["color"] = colors_cycle[i % len(colors_cycle)]

                    fig_sup = _make_line_chart(traces,
                        f"{supply_product} 공급량 — 실적 vs 예측 (R²={r2_supply:.4f})",
                        "월", "공급량 (MJ)")
                    st.plotly_chart(fig_sup, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))

                    with st.expander(f"🔎 {supply_product} — 검침기온↔공급량 산점도"):
                        fig_sc = _make_scatter_chart(x_sup_tr, y_sup_tr,
                            f"{supply_product} — 검침기온 vs 공급량", "검침기온 (℃)", "공급량 (MJ)", r2_supply)
                        st.plotly_chart(fig_sc, use_container_width=True,
                                        config=dict(scrollZoom=True, displaylogo=False))

            # ─────────────────────────────────
            # SUB 2: 판매량 기반 예측
            # ─────────────────────────────────
            with sub2:
                if not has_sales_model:
                    st.warning("판매량 학습 데이터가 부족하여 예측할 수 없습니다.")
                else:
                    st.markdown(f'<div class="sub">📦 {cooling_col} — 판매량 기반 Poly-3</div>',
                                unsafe_allow_html=True)
                    st.caption(f"Train R² = {r2_sales:.4f} | {poly_eq_text(model_sales)}")

                    tbl_sal = pd.DataFrame({"월": [f"{m}월" for m in pred_months]})
                    tbl_sal["검침기온"] = pred_temps
                    tbl_sal["예측_판매량"] = np.nan
                    j = 0
                    for i, v in enumerate(valid_mask):
                        if v:
                            tbl_sal.loc[i, "예측_판매량"] = int(pred_sales_A[j])
                            j += 1

                    if not actual_sales_yr.empty:
                        act_s = actual_sales_yr.set_index("월")[cooling_col]
                        tbl_sal["실적_판매량"] = [act_s.get(m, np.nan) for m in pred_months]
                        tbl_sal["차이"] = pd.to_numeric(tbl_sal["예측_판매량"], errors="coerce") - pd.to_numeric(tbl_sal["실적_판매량"], errors="coerce")

                    sum_row = {"월": "합계", "검침기온": ""}
                    for c in tbl_sal.columns:
                        if c not in ["월", "검침기온"]:
                            sum_row[c] = pd.to_numeric(tbl_sal[c], errors="coerce").sum()
                    tbl_sal_full = pd.concat([tbl_sal, pd.DataFrame([sum_row])], ignore_index=True)
                    int_c = [c for c in tbl_sal_full.columns if c not in ["월", "검침기온"]]
                    render_centered_table(tbl_sal_full, float_cols=["검침기온"], int_cols=int_c)

                    traces = []
                    for y in sorted(sales_years)[-3:]:
                        act = train_sales[train_sales["연"] == y].sort_values("월")
                        if not act.empty:
                            traces.append(dict(
                                x=[f"{int(m)}월" for m in act["월"]], y=act[cooling_col].values,
                                name=f"{y} 실적", color=None,
                                customdata=np.round(act["검침기온"].values, 1),
                                hover_extra="<br>검침기온 %{customdata:.1f}℃"))
                    traces.append(dict(
                        x=[f"{m}월" for m in valid_months], y=pred_sales_A,
                        name=f"예측 {cooling_pred_year}", color="#e8501a", dash="dash",
                        customdata=np.round(x_valid, 1),
                        hover_extra="<br>검침기온 %{customdata:.1f}℃"))
                    colors_cycle = ["#2563eb", "#06b6d4", "#f59e0b", "#8b5cf6"]
                    for i, t in enumerate(traces):
                        if t.get("color") is None:
                            t["color"] = colors_cycle[i % len(colors_cycle)]

                    fig_sal = _make_line_chart(traces,
                        f"{cooling_col} 판매량 — 실적 vs 예측 (R²={r2_sales:.4f})",
                        "월", "판매량 (GJ)")
                    st.plotly_chart(fig_sal, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))

                    with st.expander(f"🔎 {cooling_col} — 검침기온↔판매량 산점도"):
                        fig_sc = _make_scatter_chart(x_s_tr, y_s_tr,
                            f"{cooling_col} — 검침기온 vs 판매량", "검침기온 (℃)", "판매량 (GJ)", r2_sales)
                        st.plotly_chart(fig_sc, use_container_width=True,
                                        config=dict(scrollZoom=True, displaylogo=False))

            # ─────────────────────────────────
            # SUB 3: 실적 vs 예측 비교
            # ─────────────────────────────────
            with sub3:
                st.markdown(f'<div class="sub">📊 {cooling_pred_year}년 — 공급량 vs 판매량 예측 비교</div>',
                            unsafe_allow_html=True)

                # 비교 테이블
                cmp = pd.DataFrame({"월": [f"{m}월" for m in pred_months]})
                cmp["검침기온"] = pred_temps

                if has_supply_model:
                    col_sp = "예측_공급량"
                    cmp[col_sp] = np.nan
                    j = 0
                    for i, v in enumerate(valid_mask):
                        if v:
                            cmp.loc[i, col_sp] = int(pred_supply_A[j]); j += 1

                if has_sales_model:
                    col_sl = "예측_판매량"
                    cmp[col_sl] = np.nan
                    j = 0
                    for i, v in enumerate(valid_mask):
                        if v:
                            cmp.loc[i, col_sl] = int(pred_sales_A[j]); j += 1

                # 실적 열
                if not actual_supply_yr.empty and has_supply_model:
                    act_sup = actual_supply_yr.set_index("월")[supply_product]
                    cmp["실적_공급량"] = [act_sup.get(m, np.nan) for m in pred_months]
                if not actual_sales_yr.empty and has_sales_model:
                    act_sal = actual_sales_yr.set_index("월")[cooling_col]
                    cmp["실적_판매량"] = [act_sal.get(m, np.nan) for m in pred_months]

                sum_row = {"월": "합계", "검침기온": ""}
                for c in cmp.columns:
                    if c not in ["월", "검침기온"]:
                        sum_row[c] = pd.to_numeric(cmp[c], errors="coerce").sum()
                cmp_full = pd.concat([cmp, pd.DataFrame([sum_row])], ignore_index=True)
                int_c = [c for c in cmp_full.columns if c not in ["월", "검침기온"]]
                render_centered_table(cmp_full, float_cols=["검침기온"], int_cols=int_c)

                # 비교 라인 차트
                traces = []
                if has_supply_model:
                    traces.append(dict(x=[f"{m}월" for m in valid_months], y=pred_supply_A,
                        name="예측 공급량", color="#2563eb", dash="dash"))
                if has_sales_model:
                    traces.append(dict(x=[f"{m}월" for m in valid_months], y=pred_sales_A,
                        name="예측 판매량", color="#e8501a", dash="dash"))
                if not actual_supply_yr.empty:
                    act = actual_supply_yr.sort_values("월")
                    traces.append(dict(x=[f"{int(m)}월" for m in act["월"]],
                        y=act[supply_product].values,
                        name="실적 공급량", color="#2563eb"))
                if not actual_sales_yr.empty:
                    act = actual_sales_yr.sort_values("월")
                    traces.append(dict(x=[f"{int(m)}월" for m in act["월"]],
                        y=act[cooling_col].values,
                        name="실적 판매량", color="#16a34a"))

                if traces:
                    fig_cmp = _make_line_chart(traces,
                        f"{cooling_pred_year}년 — 공급량 vs 판매량 비교", "월", "값 (MJ/GJ)", 450)
                    st.plotly_chart(fig_cmp, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))

                # R² 요약 카드
                st.markdown("---")
                st.markdown('<div class="sub">📌 모델 정확도 요약</div>', unsafe_allow_html=True)

                def _calc_r2_vs_actual(predictions, actuals_df, target_col, valid_m):
                    """예측 vs 실적 R² 계산"""
                    if predictions is None or actuals_df.empty:
                        return np.nan
                    act_by_m = actuals_df.set_index("월")[target_col]
                    pairs = []
                    for i, m in enumerate(valid_m):
                        a = act_by_m.get(m, np.nan)
                        if not np.isnan(a) and i < len(predictions):
                            pairs.append((predictions[i], float(a)))
                    if len(pairs) < 3:
                        return np.nan
                    pred_arr = np.array([p[0] for p in pairs])
                    act_arr  = np.array([p[1] for p in pairs])
                    ss_res = np.sum((act_arr - pred_arr)**2)
                    ss_tot = np.sum((act_arr - np.mean(act_arr))**2)
                    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                r2_sup_actual = _calc_r2_vs_actual(pred_supply_A, actual_supply_yr, supply_product, valid_months) if has_supply_model else np.nan
                r2_sal_actual = _calc_r2_vs_actual(pred_sales_A, actual_sales_yr, cooling_col, valid_months) if has_sales_model else np.nan

                cc1, cc2 = st.columns(2)
                with cc1:
                    if has_supply_model:
                        r2_txt = f"{r2_sup_actual:.4f}" if not np.isnan(r2_sup_actual) else "실적 없음"
                        st.metric("공급량 기반 모델", f"Train R² = {r2_supply:.4f}",
                                  delta=f"Pred R² = {r2_txt}")
                with cc2:
                    if has_sales_model:
                        r2_txt = f"{r2_sal_actual:.4f}" if not np.isnan(r2_sal_actual) else "실적 없음"
                        st.metric("판매량 기반 모델", f"Train R² = {r2_sales:.4f}",
                                  delta=f"Pred R² = {r2_txt}")

            # ─────────────────────────────────
            # SUB 4: 기온방식 비교
            # ─────────────────────────────────
            with sub4:
                st.markdown("""
                <div class="info-box">
                <b>방식 A (기간평균)</b>: 검침기간(~30일) 기온을 평균 → 하나의 기온값으로 Poly-3 예측<br>
                <b>방식 B (일별합산)</b>: 검침기간 각 일별 기온으로 Poly-3 예측 → 일별 예측값의 평균<br>
                <b>원리</b>: Poly-3는 비선형(3차)이므로 f(평균x) ≠ 평균(f(x)). 기온 변동이 클수록 차이가 발생합니다.
                </div>
                """, unsafe_allow_html=True)

                # 학습 데이터 기간에 대해 두 방식의 예측을 비교
                st.markdown('<div class="sub">📊 학습기간 역예측(back-test) R² 비교</div>',
                            unsafe_allow_html=True)
                st.caption("학습에 사용된 각 월에 대해 방식A·B 예측값을 계산하고, 실제값과 비교합니다.")

                comparison_rows = []

                # 판매량 모델
                if has_sales_model:
                    preds_A_sales = []
                    preds_B_sales = []
                    actuals_sales = []
                    for _, row in train_sales.iterrows():
                        y_r, m_r = int(row["연"]), int(row["월"])
                        actual_v = float(row[cooling_col])
                        # Method A
                        avg_t = row["검침기온"]
                        pA = model_sales.predict(poly_sales.transform([[avg_t]]))[0]
                        # Method B
                        dtk = daily_temps_dict.get((y_r, m_r))
                        if dtk is not None and len(dtk) > 0:
                            pB = predict_daily_avg(model_sales, poly_sales, dtk)
                        else:
                            pB = pA  # fallback
                        preds_A_sales.append(pA)
                        preds_B_sales.append(pB)
                        actuals_sales.append(actual_v)

                    pA_arr = np.array(preds_A_sales)
                    pB_arr = np.array(preds_B_sales)
                    act_arr = np.array(actuals_sales)

                    def _r2(pred, act):
                        ss_res = np.sum((act - pred)**2)
                        ss_tot = np.sum((act - np.mean(act))**2)
                        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                    r2_A_sales = _r2(pA_arr, act_arr)
                    r2_B_sales = _r2(pB_arr, act_arr)
                    mae_A_sales = np.mean(np.abs(act_arr - pA_arr))
                    mae_B_sales = np.mean(np.abs(act_arr - pB_arr))

                    comparison_rows.append({
                        "모델": "판매량 기반",
                        "방식A R²": f"{r2_A_sales:.4f}",
                        "방식B R²": f"{r2_B_sales:.4f}",
                        "방식A MAE": f"{mae_A_sales:,.0f}",
                        "방식B MAE": f"{mae_B_sales:,.0f}",
                        "유리한 방식": "방식B (일별)" if r2_B_sales > r2_A_sales else "방식A (평균)",
                    })

                # 공급량 모델
                if has_supply_model:
                    preds_A_sup = []
                    preds_B_sup = []
                    actuals_sup = []
                    for _, row in train_supply.iterrows():
                        y_r, m_r = int(row["연"]), int(row["월"])
                        actual_v = float(row[supply_product])
                        avg_t = row["검침기온"]
                        pA = model_supply.predict(poly_supply.transform([[avg_t]]))[0]
                        dtk = daily_temps_dict.get((y_r, m_r))
                        if dtk is not None and len(dtk) > 0:
                            pB = predict_daily_avg(model_supply, poly_supply, dtk)
                        else:
                            pB = pA
                        preds_A_sup.append(pA)
                        preds_B_sup.append(pB)
                        actuals_sup.append(actual_v)

                    pA_arr = np.array(preds_A_sup)
                    pB_arr = np.array(preds_B_sup)
                    act_arr = np.array(actuals_sup)

                    r2_A_sup = _r2(pA_arr, act_arr)
                    r2_B_sup = _r2(pB_arr, act_arr)
                    mae_A_sup = np.mean(np.abs(act_arr - pA_arr))
                    mae_B_sup = np.mean(np.abs(act_arr - pB_arr))

                    comparison_rows.append({
                        "모델": "공급량 기반",
                        "방식A R²": f"{r2_A_sup:.4f}",
                        "방식B R²": f"{r2_B_sup:.4f}",
                        "방식A MAE": f"{mae_A_sup:,.0f}",
                        "방식B MAE": f"{mae_B_sup:,.0f}",
                        "유리한 방식": "방식B (일별)" if r2_B_sup > r2_A_sup else "방식A (평균)",
                    })

                if comparison_rows:
                    cmp_df = pd.DataFrame(comparison_rows)
                    render_centered_table(cmp_df)
                else:
                    st.warning("비교할 모델이 없습니다.")

                # 월별 상세 차이
                st.markdown("---")
                st.markdown(f'<div class="sub">📋 {cooling_pred_year}년 월별 방식 비교</div>',
                            unsafe_allow_html=True)
                st.caption("방식A(기간평균)와 방식B(일별합산)의 월별 예측값 차이")

                detail = pd.DataFrame({"월": [f"{m}월" for m in valid_months]})
                if has_supply_model and pred_supply_A is not None:
                    detail["공급량_A(평균)"] = pred_supply_A
                    clean_B = np.where(np.isnan(pred_supply_B[:len(valid_months)]), 0,
                                       pred_supply_B[:len(valid_months)])
                    detail["공급량_B(일별)"] = np.clip(np.rint(clean_B).astype(np.int64), 0, None)
                    detail["공급량_차이(B-A)"] = detail["공급량_B(일별)"].values - pred_supply_A.astype(np.int64)

                if has_sales_model and pred_sales_A is not None:
                    detail["판매량_A(평균)"] = pred_sales_A
                    clean_B = np.where(np.isnan(pred_sales_B[:len(valid_months)]), 0,
                                       pred_sales_B[:len(valid_months)])
                    detail["판매량_B(일별)"] = np.clip(np.rint(clean_B).astype(np.int64), 0, None)
                    detail["판매량_차이(B-A)"] = detail["판매량_B(일별)"].values - pred_sales_A.astype(np.int64)

                sum_row = {"월": "합계"}
                for c in detail.columns:
                    if c != "월":
                        sum_row[c] = pd.to_numeric(detail[c], errors="coerce").sum()
                detail_full = pd.concat([detail, pd.DataFrame([sum_row])], ignore_index=True)
                int_c = [c for c in detail_full.columns if c != "월"]
                render_centered_table(detail_full, int_cols=int_c)

                # 방식 비교 차트 (판매량 기준)
                if has_sales_model and pred_sales_A is not None:
                    st.markdown("---")
                    traces = [
                        dict(x=[f"{m}월" for m in valid_months], y=pred_sales_A,
                             name="방식A (기간평균)", color="#2563eb"),
                    ]
                    clean_B = np.where(np.isnan(pred_sales_B[:len(valid_months)]), 0,
                                       pred_sales_B[:len(valid_months)])
                    traces.append(
                        dict(x=[f"{m}월" for m in valid_months],
                             y=np.clip(np.rint(clean_B).astype(np.int64), 0, None),
                             name="방식B (일별합산)", color="#16a34a", dash="dashdot"))
                    if not actual_sales_yr.empty:
                        act = actual_sales_yr.sort_values("월")
                        traces.append(dict(x=[f"{int(m)}월" for m in act["월"]],
                            y=act[cooling_col].values,
                            name="실적", color="#f59e0b", width=3))

                    fig_method = _make_line_chart(traces,
                        f"판매량 기온방식 비교 — {cooling_pred_year}년",
                        "월", "판매량", 420)
                    st.plotly_chart(fig_method, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))

            # ── 엑셀 다운로드 ──
            st.markdown("---")
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                if has_supply_model:
                    tbl_sup_full.to_excel(writer, sheet_name="공급량기반_예측", index=False)
                if has_sales_model:
                    tbl_sal_full.to_excel(writer, sheet_name="판매량기반_예측", index=False)
                if comparison_rows:
                    cmp_df.to_excel(writer, sheet_name="기온방식비교", index=False)
                detail_full.to_excel(writer, sheet_name="월별방식비교", index=False)
            st.download_button("⬇️ 냉난방공조용 예측 엑셀 다운로드", data=buf.getvalue(),
                file_name=f"냉난방공조용_예측_{cooling_pred_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _parse_uploaded_temp(uploaded_file):
    try:
        name = getattr(uploaded_file, "name", "")
        if name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        if "연" not in df.columns and "월" not in df.columns:
            date_col = None
            for c in df.columns:
                if c in ["날짜", "일자", "date", "Date"]:
                    date_col = c; break
            if date_col is None:
                date_col = df.columns[0]
            df["날짜"] = pd.to_datetime(df[date_col], errors="coerce")
            df["연"] = df["날짜"].dt.year
            df["월"] = df["날짜"].dt.month
        temp_col = None
        for c in df.columns:
            if "예상" in c or "평균기온" in c or "기온" in c or "temp" in c.lower():
                temp_col = c; break
        if temp_col is None:
            for c in df.columns:
                if c not in ["연", "월", "날짜", "일자"] and pd.api.types.is_numeric_dtype(df[c]):
                    temp_col = c; break
        if temp_col is None:
            st.sidebar.error("예상기온 파일에서 기온 열을 찾지 못했습니다.")
            return None
        result = pd.DataFrame({
            "연": df["연"].astype(int), "월": df["월"].astype(int),
            "예상기온": pd.to_numeric(df[temp_col], errors="coerce"),
        }).dropna()
        return result
    except Exception as e:
        st.sidebar.error(f"예상기온 파일 파싱 실패: {e}")
        return None


if __name__ == "__main__":
    main()
