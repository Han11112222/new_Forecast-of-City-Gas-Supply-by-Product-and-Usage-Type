# app.py — 도시가스 공급량·판매량 예측
# Tab 1: 학습 기간 추천 (기온 학습 기간 + Poly-3 학습 기간)
# Tab 2: 공급량 예측 (Poly-3 + Normal/Best/Conservative)
# Tab 3: 판매량 예측 (냉방용) — 동절기/하절기 분리 모델(Ver1/Ver2/Ver3) + 계획 비교
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

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
    x_pred = np.asarray(x_pred, dtype=float)
    if len(x_tr) < 4:
        return np.full_like(x_pred, np.nan, dtype=float), 0.0, None, None
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_tr.reshape(-1, 1))
    model = LinearRegression().fit(Xtr, y_tr)
    r2 = model.score(Xtr, y_tr)
    # x_pred에 NaN이 섞여 있어도 sklearn이 전체를 거부하며 죽지 않도록,
    # 유효한 값만 모델에 넣고 나머지는 NaN으로 채운다.
    y_pred = np.full_like(x_pred, np.nan, dtype=float)
    valid = ~np.isnan(x_pred)
    if valid.any():
        y_pred[valid] = model.predict(poly.transform(x_pred[valid].reshape(-1, 1)))
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
# 냉방용 사용량 분석 심화ver — Tab 3 전용 (동절기/하절기 분리 모델, 판매량 계획 비교)
# ══════════════════════════════════════════════

LINE_COLORS = {
    '실제_공급량합계':     "#1f4e9c",
    '방법1_예측(정밀)':    "#2ecc71",
    '방법2_예측(단순)':    "#f39c12",
    '판매량_실적':       "#dc2626",
    '예측_판매량_v1':         "#66b2ff",
    '예측_판매량_v2':      "#f39c12",
    '예측_판매량_v3':      "#8e44ad",
    '판매량_계획':         "#f1948a",
    '검침기온':           "#059669",
}

SERIES_LABELS = {
    '판매량_실적':  '실적',
    '예측_판매량_v1':    '기존 단일 3차식',
    '예측_판매량_v2': '분리·3차식(참고)',
    '예측_판매량_v3': '분리·2차식',
    '판매량_계획':    '판매량 계획',
}



def render_line_chart(df, x_col, y_cols, height=420, title=None,
                      secondary_col=None, secondary_name=None, secondary_suffix="℃"):
    """
    범례를 클릭하면 해당 라인을 껐다 켰다 할 수 있는 인터랙티브 라인차트.
    df: x_col을 포함한 DataFrame (set_index 하지 않은 상태로 전달)
    y_cols: 그릴 컬럼 이름 리스트 (df에 없는 컬럼은 자동으로 건너뜀)
    secondary_col: 우측 보조축(예: 기온)에 점선으로 추가할 컬럼 (선택)
    """
    fig = go.Figure()
    for col in y_cols:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], mode="lines+markers", name=col,
            line=dict(color=LINE_COLORS.get(col), width=2.2),
            marker=dict(size=5),
        ))
    has_secondary = secondary_col is not None and secondary_col in df.columns
    if has_secondary:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[secondary_col], mode="lines+markers",
            name=secondary_name or secondary_col,
            line=dict(color=LINE_COLORS.get(secondary_col, "#059669"), width=2, dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            yaxis="y2",
        ))
    layout_kwargs = dict(
        height=height,
        margin=dict(t=40 if title else 10, b=10, l=50, r=50 if has_secondary else 20),
        hovermode="x unified",
        yaxis=dict(rangemode="tozero", tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    if has_secondary:
        layout_kwargs["yaxis2"] = dict(
            overlaying="y", side="right", showgrid=False,
            ticksuffix=secondary_suffix, title=None,
        )
    if title:  # title=None을 그대로 넘기면 프론트엔드에서 "undefined"로 표시되는 문제 방지
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False))


def render_r2_mae_card(col, label, r2, mae, delta_r2=None):
    """R²(위)와 MAE(아래)를 세로 배치하는 카드. 모든 카드에서 통일된 레이아웃."""
    delta_html = ""
    if delta_r2 is not None:
        color = "#16a34a" if delta_r2 >= 0 else "#dc2626"
        arrow = "↑" if delta_r2 >= 0 else "↓"
        sign = "+" if delta_r2 >= 0 else ""
        delta_html = (f'<span style="font-size:0.82rem;color:{color};margin-left:0.4rem;">'
                      f'{arrow} {sign}{delta_r2:.4f}</span>')
    col.markdown(f"""
<div style="font-size:0.8rem;color:#666;margin-bottom:2px;">{label}</div>
<div style="font-size:1.9rem;font-weight:700;color:#1f2937;line-height:1.2;">
  {r2:.4f}{delta_html}
</div>
<div style="margin-top:4px;">
  <span style="font-size:1.4rem;font-weight:700;color:#166534;background-color:#dcfce7;
               padding:0.1em 0.5em;border-radius:0.4em;">MAE {mae:,.0f}</span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 공통: 방법2용 구글시트 일별 평균기온 → 월별 평균 집계

SALES_SHEET_URL = "https://docs.google.com/spreadsheets/d/1-8RIPIkjnVXxoh5QJs6598nnHkWOGmrO655jr3b3g04/export?format=csv&gid=0"
PLAN_SHEET_URL = "https://docs.google.com/spreadsheets/d/1zu2R21_P6z6yCeWz7yX1K6IYhj541hcr3IvCAaHLEQ8/export?format=csv&gid=0"

# ── Ver2(동절기/하절기 분리 모델)용 기준온도 — 기존 HDD/CDD 기준과 동일 ──
WINTER_T = 18.0  # 검침기온 ≤ 18℃ → 동절기 모델 (HDD 기준온도)
SUMMER_T = 26.0  # 검침기온 ≥ 26℃ → 하절기 모델 (CDD 기준온도)

@st.cache_data
def load_daily_temp_for_cooling():
    """
    구글시트(13HrIz6O...)의 일자 단위 원본 기온을 그대로 로드한다.
    (load_monthly_avg_temp()는 이미 월평균으로 뭉개버리므로,
     검침기간 전월16~당월15 계산을 위해 일자 단위로 별도 로드)
    """
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"❌ 일별기온 구글시트 로드 오류: {e}")
        st.stop()

    col_list = df.columns.tolist()
    date_cols = [c for c in col_list if '날짜' in c or 'date' in c.lower() or 'Date' in c]
    DATE_COL = date_cols[0] if date_cols else col_list[0]
    temp_cols = [c for c in col_list if '평균기온' in c] or \
                [c for c in col_list if '기온' in c or 'temp' in c.lower()]
    TEMP_COL = temp_cols[0] if temp_cols else col_list[1]

    df['Date'] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day']   = df['Date'].dt.day
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce')
    df = df.dropna(subset=[TEMP_COL])
    return df[['Date', 'Year', 'Month', 'Day', TEMP_COL]].rename(columns={TEMP_COL: 'Avg_Temp'})


def compute_meter_reading_temp(daily_df):
    """
    검침기간 평균기온 = 전월16일~말일 + 당월1일~15일 평균 (일평균기온 기준, 정밀 시간대 아님).
    반환: DataFrame(Year, Month, 검침기온)
    """
    rows = []
    for (y, m), _ in daily_df.groupby(['Year', 'Month']):
        cur_half = daily_df[(daily_df['Year'] == y) & (daily_df['Month'] == m) &
                             (daily_df['Day'] <= 15)]['Avg_Temp']
        py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
        prev_half = daily_df[(daily_df['Year'] == py) & (daily_df['Month'] == pm) &
                              (daily_df['Day'] >= 16)]['Avg_Temp']
        combined = pd.concat([prev_half, cur_half]).dropna()
        if len(combined) >= 5:
            rows.append({'Year': int(y), 'Month': int(m), '검침기온': combined.mean()})
    return pd.DataFrame(rows)


@st.cache_data
def load_cooling_sales():
    """판매량 실적 구글시트 — '냉방용' 컬럼 로드."""
    try:
        df = pd.read_csv(SALES_SHEET_URL)
    except Exception as e:
        st.error(f"❌ 판매량 구글시트 로드 오류: {e}")
        st.stop()

    col_list = df.columns.tolist()
    cooling_col = None
    for c in col_list:
        if '냉방' in c:
            cooling_col = c; break
    if cooling_col is None:
        st.error("판매량 시트에서 '냉방용' 컬럼을 찾을 수 없습니다.")
        st.stop()

    year_col  = '연' if '연' in col_list else ('Year' if 'Year' in col_list else col_list[1])
    month_col = '월' if '월' in col_list else ('Month' if 'Month' in col_list else col_list[2])

    out = df.rename(columns={year_col: 'Year', month_col: 'Month'})[['Year', 'Month', cooling_col]].copy()
    out[cooling_col] = pd.to_numeric(
        out[cooling_col].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')
    out['Year']  = pd.to_numeric(out['Year'], errors='coerce')
    out['Month'] = pd.to_numeric(out['Month'], errors='coerce')
    out = out.dropna(subset=['Year', 'Month', cooling_col])
    out['Year']  = out['Year'].astype(int)
    out['Month'] = out['Month'].astype(int)
    out = out[out[cooling_col] > 0].reset_index(drop=True)
    return out.rename(columns={cooling_col: '판매량_실적'})


@st.cache_data
def load_cooling_plan():
    """
    '상품별판매량 계획' 구글시트 — '냉방용' 컬럼(기존 계획값) 로드.
    로드 실패/컬럼 미탐지 시 None을 반환하며, 호출부에서 계획 비교 없이 진행하도록 처리한다.
    """
    try:
        df = pd.read_csv(PLAN_SHEET_URL)
    except Exception as e:
        st.sidebar.warning(f"⚠️ 판매량 계획 시트 로드 실패: {e} (계획 비교 생략)")
        return None

    col_list = df.columns.tolist()
    plan_col = None
    for c in col_list:
        if '냉방' in c:
            plan_col = c; break
    if plan_col is None:
        st.sidebar.warning("판매량 계획 시트에서 '냉방용' 컬럼을 찾을 수 없어 계획 비교를 생략합니다.")
        return None

    year_col  = '연' if '연' in col_list else ('Year' if 'Year' in col_list else col_list[1])
    month_col = '월' if '월' in col_list else ('Month' if 'Month' in col_list else col_list[2])

    out = df.rename(columns={year_col: 'Year', month_col: 'Month'})[['Year', 'Month', plan_col]].copy()
    out[plan_col] = pd.to_numeric(
        out[plan_col].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')
    out['Year']  = pd.to_numeric(out['Year'], errors='coerce')
    out['Month'] = pd.to_numeric(out['Month'], errors='coerce')
    out = out.dropna(subset=['Year', 'Month', plan_col])
    out['Year']  = out['Year'].astype(int)
    out['Month'] = out['Month'].astype(int)
    return out.rename(columns={plan_col: '판매량_계획'})


def fit_piecewise_seasonal_models(train_df, x_col='검침기온', y_col='판매량_실적', degree=3):
    """
    검침기온 기준 동절기(≤WINTER_T)/하절기(≥SUMMER_T) 데이터를 각각 나눠
    별도의 다항식 모델을 학습한다. (이중계상 방지를 위해 중간구간 데이터는 학습에서 제외,
    예측 시 18℃/26℃ 경계값을 선형보간하여 연결)
    """
    winter_data = train_df[train_df[x_col] <= WINTER_T]
    summer_data = train_df[train_df[x_col] >= SUMMER_T]

    models = {'winter': None, 'summer': None}
    min_pts = degree + 1
    if len(winter_data) >= min_pts:
        mw = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        mw.fit(winter_data[[x_col]], winter_data[y_col])
        models['winter'] = mw
    if len(summer_data) >= min_pts:
        ms = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        ms.fit(summer_data[[x_col]], summer_data[y_col])
        models['summer'] = ms
    return models, winter_data, summer_data


def predict_piecewise_seasonal(models, x_values):
    """
    x_values(검침기온 배열)에 대해:
      x <= WINTER_T        → 동절기 모델 예측
      x >= SUMMER_T         → 하절기 모델 예측
      WINTER_T < x < SUMMER_T → 두 모델의 경계값(18℃/26℃ 지점 예측)을 선형보간
    """
    x_arr = np.asarray(x_values, dtype=float)
    mw, ms = models.get('winter'), models.get('summer')
    w_at_boundary = float(mw.predict([[WINTER_T]])[0]) if mw is not None else None
    s_at_boundary = float(ms.predict([[SUMMER_T]])[0]) if ms is not None else None

    preds = np.full_like(x_arr, np.nan, dtype=float)
    for i, x in enumerate(x_arr):
        if np.isnan(x):
            continue
        if x <= WINTER_T:
            preds[i] = float(mw.predict([[x]])[0]) if mw is not None else np.nan
        elif x >= SUMMER_T:
            preds[i] = float(ms.predict([[x]])[0]) if ms is not None else np.nan
        else:
            if w_at_boundary is not None and s_at_boundary is not None:
                frac = (x - WINTER_T) / (SUMMER_T - WINTER_T)
                preds[i] = w_at_boundary * (1 - frac) + s_at_boundary * frac
            elif w_at_boundary is not None:
                preds[i] = w_at_boundary
            elif s_at_boundary is not None:
                preds[i] = s_at_boundary
    return preds


def poly_eq_str(coefs, intercept):
    """
    PolynomialFeatures(degree=n, include_bias=False) 계수 배열(coefs, 오름차순: x, x², x³...)과
    절편(intercept)을 받아 차수에 상관없이 "y = ax^n + ... + c" 형태 문자열을 만든다.
    """
    n = len(coefs)
    parts = []
    for power in range(n, 0, -1):
        c = coefs[power - 1]
        parts.append(f"{c:+.2f}x^{power}" if power > 1 else f"{c:+.2f}x")
    parts.append(f"{intercept:+.0f}")
    eq = " ".join(parts)
    if eq.startswith("+"):
        eq = eq[1:]
    return f"y = {eq}"


def _dynamic_fmt(df, x_col):
    """df의 x_col을 제외한 모든 컬럼에 대해 포맷을 자동 결정한다.
    '오차율' 또는 'MAPE'가 들어간 컬럼은 %, '기온'이 들어간 컬럼은 소수 1자리+℃, 나머지는 천단위 콤마."""
    fmt = {}
    for c in df.columns:
        if c == x_col:
            continue
        if '오차율' in c or 'MAPE' in c:
            fmt[c] = "{:.1f}%"
        elif '기온' in c:
            fmt[c] = "{:.1f}℃"
        else:
            fmt[c] = "{:,.0f}"
    return fmt


def _apply_mae_toggle(df, x_col, use_abs):
    """use_abs=True면 '차이'/'오차율' 컬럼을 절대값으로 바꾸고, '차이'→'MAE', '오차율(%)'→'MAPE(%)'로 표시한다."""
    if not use_abs:
        return df
    out = df.copy()
    rename_map = {}
    for c in out.columns:
        if c == x_col:
            continue
        if '차이' in c:
            out[c] = out[c].abs()
            rename_map[c] = c.replace('차이', 'MAE')
        elif '오차율' in c:
            out[c] = out[c].abs()
            rename_map[c] = c.replace('오차율(%)', 'MAPE(%)')
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _ensure_baseline_cols(selected, target_col, plan_col='판매량_계획', has_plan=False):
    """
    표에는 사용자가 멀티선택에서 빼더라도 '계획'과 '실적'(target_col)을 항상 맨 앞에 강제로 포함시킨다.
    (이 둘이 빠지면 비교 기준이 없어져 차이/하이라이트가 전혀 표시되지 않기 때문)
    순서: [판매량_계획(있으면), target_col, 그 외 선택된 예측 시리즈(선택 순서 유지)]
    """
    others = [c for c in selected if c not in (plan_col, target_col)]
    result = [plan_col] if has_plan else []
    result.append(target_col)
    result += others
    return result


_DIFF_TABLE_CSS = """
<style>
.difftbl-wrap { overflow-x:auto; border:1px solid #e2e8f0; border-radius:6px; margin-bottom:0.8rem; }
.difftbl { border-collapse:collapse; font-size:0.82rem;
           font-family:'Segoe UI','Noto Sans KR',sans-serif; }
.difftbl th {
    background:#f8fafc; color:#334155; padding:6px 10px; text-align:center;
    border:1px solid #e2e8f0; font-weight:600; white-space:normal;
    word-break:keep-all; min-width:120px; max-width:150px; line-height:1.3;
}
.difftbl td { padding:5px 10px; text-align:right; border:1px solid #eef1f5; white-space:nowrap; }
.difftbl th.difftbl-x, .difftbl td.difftbl-x { background:#eef2f7 !important; font-weight:600; }
.difftbl td.difftbl-x { text-align:center; }
.difftbl th.difftbl-target, .difftbl td.difftbl-target { background:#dbeafe !important; font-weight:600; }
.difftbl tr:hover td { background:#f8fafc; }
</style>
"""


def _fmt_diff_value(col, val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if '오차율' in col or 'MAPE' in col:
        return f"{val:.1f}%"
    if '기온' in col:
        return f"{val:.1f}℃"
    try:
        return f"{val:,.0f}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_x_value(val):
    """구분 열(Year 등) 표시용. 정수형 float(예: 2023.0)이면 '.0'을 떼고 정수로 보여준다."""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def render_html_diff_table(df, x_col, target_col=None):
    """
    표를 HTML 테이블로 렌더링한다 (st.dataframe은 헤더 줄바꿈을 지원하지 않아 텍스트가
    잘리는 문제가 있어, 컬럼명의 '\\n'을 <br>로 바꿔 풀네임을 2줄로 보여주기 위함).
    x_col(구분 열)과 target_col(실적 등 기준 열)은 배경색으로 하이라이트한다.
    """
    cols = list(df.columns)

    def _cls(c):
        if c == x_col:
            return ' class="difftbl-x"'
        if target_col and c == target_col:
            return ' class="difftbl-target"'
        return ""

    hdr = "".join(f"<th{_cls(c)}>{c.replace(chr(10), '<br>')}</th>" for c in cols)
    body = ""
    for _, row in df.iterrows():
        cells = "".join(
            f"<td{_cls(c)}>{_fmt_x_value(row[c]) if c == x_col else _fmt_diff_value(c, row[c])}</td>" for c in cols)
        body += f"<tr>{cells}</tr>"

    st.markdown(f"""{_DIFF_TABLE_CSS}
<div class="difftbl-wrap">
<table class="difftbl">
<thead><tr>{hdr}</tr></thead>
<tbody>{body}</tbody>
</table>
</div>""", unsafe_allow_html=True)


def render_diff_table(df, x_col, target_col=None, key_prefix="tbl"):
    """
    비교표 렌더링 공통 헬퍼.
    - 좌측 상단에 'MAE 변환' 토글(체크박스)을 두고, 켜면 차이 컬럼을 절대값(MAE 스타일)으로 표시
    - x_col(구분 열)과 target_col(실적 등 기준 열)에 배경색 하이라이트 적용
    - 컬럼명에 줄바꿈(\\n)이 들어간 긴 헤더는 HTML 테이블로 렌더링해 풀네임을 2줄로 보여준다.
    """
    use_mae = st.checkbox("📌 차이를 절대값(MAE)으로 표시", key=f"{key_prefix}_mae_toggle")
    disp = _apply_mae_toggle(df, x_col, use_mae)
    render_html_diff_table(disp, x_col, target_col=target_col)


def render_yearly_diff_table(monthly_raw_df, target_col, selected_cols, key_prefix="tbl", target_label=None):
    """
    연도별 집계표 전용 렌더러. monthly_raw_df는 'Year'(또는 'Year_Month') + 선택 시리즈의
    "월별 원본값"을 담은 DataFrame이어야 한다 (차이 컬럼 없이).

    MAE 토글 off: 연간 합계끼리의 순차이 (연간계획합 − 연간실적합) — 부호 있는 순차이.
    MAE 토글 on : 월별로 먼저 |차이|를 구한 뒤 연도별 평균 — 진짜 MAE(평균절대오차).
                  (연간 합계끼리의 차이에 단순히 절대값만 씌우면 +/-가 서로 상쇄된 순차이의
                  절대값이 나와서 실제 월별 오차 크기를 반영하지 못하므로, 반드시 월 단위에서
                  먼저 절대값을 취하고 나서 연도로 집계해야 한다.)
    """
    use_mae = st.checkbox("📌 차이를 절대값(MAE)으로 표시 — 월별 오차를 먼저 절대값화한 뒤 연평균",
                          key=f"{key_prefix}_mae_toggle")

    tmp = monthly_raw_df.copy()
    if 'Year' not in tmp.columns:
        tmp['Year'] = tmp['Year_Month'].str[:4].astype(int)

    cols = [c for c in selected_cols if c in tmp.columns]
    has_target = target_col in cols
    label = target_label or SERIES_LABELS.get(target_col, target_col)

    yearly_raw = tmp.groupby('Year')[cols].sum().reset_index()

    # MAE 모드용: 월별 signed 차이/오차율을 미리 계산
    monthly_diff, monthly_pct = {}, {}
    if has_target:
        for c in cols:
            if c == target_col:
                continue
            monthly_diff[c] = tmp[c] - tmp[target_col]
            with np.errstate(divide='ignore', invalid='ignore'):
                monthly_pct[c] = np.where(tmp[target_col] != 0, monthly_diff[c] / tmp[target_col] * 100, np.nan)

    out = pd.DataFrame({'Year': yearly_raw['Year']})
    pending, target_seen = [], False

    def add_diff(c):
        if use_mae:
            out[f'{c}\n{label}대비MAE'] = pd.Series(monthly_diff[c], index=tmp.index).abs() \
                .groupby(tmp['Year']).mean().values
            out[f'{c}\n{label}대비MAPE(%)'] = pd.Series(monthly_pct[c], index=tmp.index).abs() \
                .groupby(tmp['Year']).mean().values
        else:
            diff_val = yearly_raw[c] - yearly_raw[target_col]
            out[f'{c}\n{label}대비차이'] = diff_val
            with np.errstate(divide='ignore', invalid='ignore'):
                out[f'{c}\n{label}대비오차율(%)'] = np.where(
                    yearly_raw[target_col] != 0, diff_val / yearly_raw[target_col] * 100, np.nan)

    for c in cols:
        out[c] = yearly_raw[c]
        if c == target_col:
            target_seen = True
            for pc in pending:
                add_diff(pc)
            continue
        if not has_target:
            continue
        if not target_seen:
            pending.append(c)
        else:
            add_diff(c)

    render_html_diff_table(out, 'Year', target_col=target_col)
    return out


def _build_diff_table(df, x_col, target_col, selected_cols, target_label=None):
    """
    df에서 x_col + selected_cols(원본값 컬럼)로 표를 만든다.
    컬럼 순서는 selected_cols 순서를 따르되, target_col(예: 실적)이 먼저 나온 컬럼들의
    차이/오차율은 target_col 바로 뒤로 몰아서 보여주고, target_col 이후에 나오는 컬럼들은
    (원본값 → 차이 → 오차율) 세트로 바로 이어 붙인다.
    예) selected_cols=[계획, 실적, v1, v2] → 계획, 실적, 계획_실적대비차이, 계획_실적대비오차율(%),
        v1, v1_실적대비차이, v1_실적대비오차율(%), v2, v2_실적대비차이, v2_실적대비오차율(%)
    target_label을 안 주면 SERIES_LABELS에서 target_col의 한글 라벨을 찾아 사용한다.
    """
    cols = [c for c in selected_cols if c in df.columns]
    has_target = target_col in cols
    label = target_label or SERIES_LABELS.get(target_col, target_col)

    out = pd.DataFrame({x_col: df[x_col]})
    pending_before_target = []  # target보다 먼저 선택된 비교 대상 컬럼 (차이 계산을 target 등장 후로 미룸)
    target_seen = False

    def _add_diff(colname):
        out[f'{colname}\n{label}대비차이'] = df[colname] - df[target_col]
        with np.errstate(divide='ignore', invalid='ignore'):
            out[f'{colname}\n{label}대비오차율(%)'] = np.where(
                df[target_col] != 0, out[f'{colname}\n{label}대비차이'] / df[target_col] * 100, np.nan)

    for c in cols:
        out[c] = df[c]
        if c == target_col:
            target_seen = True
            for pc in pending_before_target:
                _add_diff(pc)
            continue
        if not has_target:
            continue
        if not target_seen:
            pending_before_target.append(c)
        else:
            _add_diff(c)
    return out


def render_cooling_analysis():
    st.header("🧊 냉방용 사용량 분석 심화ver")
    st.markdown("- 전월16일부터 당월15일까지의 실제기온 평균 적용 (세 가지 모델 모두 공통)")

    with st.spinner("냉방용 데이터를 불러오는 중입니다..."):
        daily_temp_df = load_daily_temp_for_cooling()
        meter_temp_df = compute_meter_reading_temp(daily_temp_df)
        sales_df = load_cooling_sales()
        plan_df = load_cooling_plan()  # None일 수 있음 (로드 실패/컬럼 미탐지 시 계획 비교 생략)
        merged_cool = pd.merge(meter_temp_df, sales_df, on=['Year', 'Month'], how='inner')
        merged_cool['Year_Month'] = merged_cool.apply(
            lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1)

    if merged_cool.empty:
        st.warning("실제기온과 판매량 데이터의 겹치는 기간이 없습니다.")
        st.stop()

    TARGET = '판매량_실적'
    all_years_cool = sorted(merged_cool['Year'].unique())

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧊 냉방용 분석 설정**")
    train_years_c = st.sidebar.multiselect(
        "1. AI 학습 연도 선택 (냉방용)", options=all_years_cool,
        default=all_years_cool, key="cool_train_years")
    eval_years_c = st.sidebar.multiselect(
        "2. 과거 적합도 검증 연도 (냉방용)", options=all_years_cool,
        default=all_years_cool[-2:], key="cool_eval_years")
    max_year_c = int(merged_cool['Year'].max())
    future_years_c = st.sidebar.multiselect(
        "3. 미래 시나리오 추정 연도 (냉방용)",
        options=list(range(max_year_c + 1, max_year_c + 6)),
        default=[max_year_c + 1, max_year_c + 2], key="cool_future_years")
    y_years_c = st.sidebar.slider(
        "4. 미래 예측기온 추정 기준 (최근 Y년 평균, 냉방용)",
        min_value=1, max_value=10, value=3, step=1, key="cool_y_years")
    sim_base_years_c = list(range(max_year_c - y_years_c + 1, max_year_c + 1))

    if not train_years_c or not eval_years_c:
        st.warning("👈 좌측 패널에서 냉방용 학습/검증 연도를 선택해주세요.")
        st.stop()

    train_df_c = merged_cool[merged_cool['Year'].isin(train_years_c)]
    x_train_c = train_df_c[['검침기온']]
    y_train_c = train_df_c[TARGET]

    # 기준모델(단일 3차식) — 비교 지표용으로만 사용, 별도 섹션은 만들지 않음
    model_base = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    model_base.fit(x_train_c, y_train_c)
    cb = model_base.named_steps['linearregression'].coef_
    ib = model_base.named_steps['linearregression'].intercept_

    # 분리모델(3차식) — 2차식 채택 근거 비교용 (표/차트에는 노출하지 않고 R² 지표에만 사용)
    models_cubic, winter_data_c3, summer_data_c3 = fit_piecewise_seasonal_models(
        train_df_c, x_col='검침기온', y_col=TARGET, degree=3)
    has_cubic_split_eq = models_cubic['winter'] is not None and models_cubic['summer'] is not None
    if has_cubic_split_eq:
        cw3 = models_cubic['winter'].named_steps['linearregression'].coef_
        iw3 = models_cubic['winter'].named_steps['linearregression'].intercept_
        cs3 = models_cubic['summer'].named_steps['linearregression'].coef_
        is3 = models_cubic['summer'].named_steps['linearregression'].intercept_

    # ══════════════════════════════════════════
    # 모델 설명 (요약)
    # ══════════════════════════════════════════
    st.markdown("---")
    item2_eq = ""
    if has_cubic_split_eq:
        item2_eq = f"동절기: ${poly_eq_str(cw3, iw3)}$  \n하절기: ${poly_eq_str(cs3, is3)}$"
    st.markdown(f"""
**1. 일반적인 3차 다항식 적용**
(여름, 겨울철 패턴 학습시 과대예측 발생 가능)
${poly_eq_str(cb, ib)}$

**2. 동절기/하절기 분리 (HDD {WINTER_T:.0f}℃ / CDD {SUMMER_T:.0f}℃ 기준온도 참고)**
{item2_eq}

**3. 추가 모델 (2차식)**
하절기는 학습 표본이 적어, 3차식 계수 불안정
""")

    models_final, winter_data_f, summer_data_f = fit_piecewise_seasonal_models(
        train_df_c, x_col='검침기온', y_col=TARGET, degree=2)

    if models_final['winter'] is None or models_final['summer'] is None:
        st.warning(
            f"동절기(≤{WINTER_T:.0f}℃, n={len(winter_data_f)}) 또는 "
            f"하절기(≥{SUMMER_T:.0f}℃, n={len(summer_data_f)}) 학습 데이터가 3건 미만이라 "
            "모델을 만들 수 없습니다. 학습 연도를 늘려주세요."
        )
        st.stop()

    cw = models_final['winter'].named_steps['linearregression'].coef_
    iw = models_final['winter'].named_steps['linearregression'].intercept_
    cs = models_final['summer'].named_steps['linearregression'].coef_
    isu = models_final['summer'].named_steps['linearregression'].intercept_
    r2_w = r2_score(winter_data_f[TARGET], models_final['winter'].predict(winter_data_f[['검침기온']]))
    r2_s = r2_score(summer_data_f[TARGET], models_final['summer'].predict(summer_data_f[['검침기온']]))

    col_w, col_s = st.columns(2)
    with col_w:
        st.info(f"""
**❄️ 동절기 모델 (실제기온 ≤ {WINTER_T:.0f}℃, n={len(winter_data_f)}, 2차식)**

학습 R² = {r2_w * 100:.2f}%

${poly_eq_str(cw, iw)}$
""")
    with col_s:
        st.info(f"""
**☀️ 하절기 모델 (실제기온 ≥ {SUMMER_T:.0f}℃, n={len(summer_data_f)}, 2차식)**

학습 R² = {r2_s * 100:.2f}%

${poly_eq_str(cs, isu)}$
""")
    st.caption(f"※ {WINTER_T:.0f}℃부터 {SUMMER_T:.0f}℃ 사이 구간은 두 모델의 경계값을 선형보간하여 연결(중복계상 방지) "
               f"· 기온 소스: 구글시트 일별 기온 → 검침기간(전월16일부터 당월15일까지) 평균 · 판매량 소스: 판매량 실적 시트 — 냉방용")

    with st.expander("🔎 실제기온 ↔ 냉방용 판매량 산점도 (학습 데이터)"):
        st.scatter_chart(train_df_c.rename(columns={'검침기온': '실제기온'}), x='실제기온', y=TARGET, height=380)

    # ══════════════════════════════════════════
    # 과거 적합도 검증
    # ══════════════════════════════════════════
    st.subheader("📊 과거 모델 적합도 검증 (냉방용)")
    eval_df_c = merged_cool[merged_cool['Year'].isin(eval_years_c)].copy()
    eval_df_c['예측_판매량_v1'] = model_base.predict(eval_df_c[['검침기온']])
    eval_df_c['예측_판매량_v3'] = predict_piecewise_seasonal(models_final, eval_df_c['검침기온'].values)

    has_plan_eval = False
    if plan_df is not None:
        eval_df_c = eval_df_c.merge(plan_df, on=['Year', 'Month'], how='left')
        has_plan_eval = eval_df_c['판매량_계획'].notna().any()

    valid_eval = eval_df_c['예측_판매량_v3'].notna()
    r2_base_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량_v1'])
    mae_base_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량_v1'] - eval_df_c.loc[valid_eval, TARGET]))
    r2_final_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량_v3'])
    mae_final_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량_v3'] - eval_df_c.loc[valid_eval, TARGET]))

    has_cubic_split = models_cubic['winter'] is not None and models_cubic['summer'] is not None
    if has_cubic_split:
        eval_df_c['예측_판매량_v2'] = predict_piecewise_seasonal(models_cubic, eval_df_c['검침기온'].values)
        r2_cubic_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량_v2'])
        mae_cubic_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량_v2'] - eval_df_c.loc[valid_eval, TARGET]))

    # 기존 계획(판매량_계획) 자체도 실적과 비교해 R²/MAE 산출 — "새 예측방식이 계획보다 나은가"를 바로 보여주기 위함
    if has_plan_eval:
        valid_plan_eval = valid_eval & eval_df_c['판매량_계획'].notna()
        r2_plan_eval = r2_score(eval_df_c.loc[valid_plan_eval, TARGET], eval_df_c.loc[valid_plan_eval, '판매량_계획'])
        mae_plan_eval = np.mean(np.abs(eval_df_c.loc[valid_plan_eval, '판매량_계획'] - eval_df_c.loc[valid_plan_eval, TARGET]))

    monthly_eval_c = eval_df_c[['Year_Month', 'Year', 'Month', TARGET, '예측_판매량_v1', '예측_판매량_v3', '검침기온']].copy()
    if has_cubic_split:
        monthly_eval_c['예측_판매량_v2'] = eval_df_c['예측_판매량_v2']
    if has_plan_eval:
        monthly_eval_c['판매량_계획'] = eval_df_c['판매량_계획']

    all_series_eval = (['판매량_계획'] if has_plan_eval else []) + [TARGET, '예측_판매량_v1'] \
        + (['예측_판매량_v2'] if has_cubic_split else []) + ['예측_판매량_v3']

    # R²/MAE 카드 목록 구성 — MAE가 가장 낮은 카드에 자동으로 ✅ 표시
    metrics = []
    if has_plan_eval:
        metrics.append({"key": "plan", "label": "기존 계획(판매량_계획)", "r2": r2_plan_eval,
                        "mae": mae_plan_eval, "delta": None})
    metrics.append({"key": "base", "label": "기존 단일 3차식", "r2": r2_base_eval,
                    "mae": mae_base_eval, "delta": None})
    if has_cubic_split:
        metrics.append({"key": "cubic", "label": "분리·3차식 (참고)", "r2": r2_cubic_eval,
                        "mae": mae_cubic_eval, "delta": r2_cubic_eval - r2_base_eval})
    metrics.append({"key": "final", "label": "분리·2차식", "r2": r2_final_eval,
                    "mae": mae_final_eval, "delta": r2_final_eval - r2_base_eval})

    best_i = min(range(len(metrics)), key=lambda i: metrics[i]["mae"])
    mcols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        label = f'✅ {m["label"]}' if i == best_i else m["label"]
        render_r2_mae_card(mcols[i], label, m["r2"], m["mae"], delta_r2=m["delta"])

    # 차트는 항상 전체 시리즈 표시 — 플롯리 자체 범례 클릭으로 라인 표시/숨김
    show_temp_eval = st.checkbox("🌡️ 실제기온(전월16일부터 당월15일까지) 표시", key="eval_show_temp")
    render_line_chart(monthly_eval_c, 'Year_Month', all_series_eval, height=420,
                      secondary_col='검침기온' if show_temp_eval else None,
                      secondary_name='실제기온(℃)')

    # 아래 선택 위젯은 표(연도별/월별)에만 반영됨 (차트에는 영향 없음)
    st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영됩니다)")
    selected_eval = st.multiselect(
        "표시할 시리즈", options=all_series_eval, default=all_series_eval,
        format_func=lambda c: SERIES_LABELS.get(c, c), key="eval_series_select")
    if not selected_eval:
        st.info("표시할 항목을 1개 이상 선택해주세요. 우선 전체 항목을 표시합니다.")
        selected_eval = all_series_eval

    table_series_eval = _ensure_baseline_cols(selected_eval, TARGET, has_plan=has_plan_eval)

    st.markdown("**📆 연도별 실적 대비 차이 요약**")
    yearly_table_eval = render_yearly_diff_table(monthly_eval_c, TARGET, table_series_eval, key_prefix="eval_yearly")

    monthly_table_eval = _build_diff_table(monthly_eval_c, 'Year_Month', TARGET, table_series_eval)
    st.markdown("**🗂️ 월별 상세 비교**")
    render_diff_table(monthly_table_eval, 'Year_Month', target_col=TARGET, key_prefix="eval_monthly")

    dl_eval1, dl_eval2 = st.columns(2)
    with dl_eval1:
        csv_yearly_eval = yearly_table_eval.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 연도별 요약 다운로드", data=csv_yearly_eval,
                           file_name="냉방용_연도별요약.csv", mime="text/csv", key="dl_eval_yearly")
    with dl_eval2:
        csv_monthly_eval = monthly_table_eval.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 월별 상세 다운로드", data=csv_monthly_eval,
                           file_name="냉방용_과거적합도_검증리포트.csv", mime="text/csv", key="dl_eval_monthly")

    # ══════════════════════════════════════════
    # 미래 시나리오
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔮 미래 냉방용 판매량 추정 시나리오")

    if future_years_c:
        hist_temp_c = meter_temp_df[meter_temp_df['Year'].isin(sim_base_years_c)]
        sim_month_temp_c = hist_temp_c.groupby('Month')['검침기온'].mean().reset_index()

        future_rows = []
        for y in future_years_c:
            for m in range(1, 13):
                t = sim_month_temp_c.loc[sim_month_temp_c['Month'] == m, '검침기온']
                if len(t) > 0:
                    future_rows.append({'Year': y, 'Month': m, '검침기온': float(t.values[0])})
        future_df_c = pd.DataFrame(future_rows)
        future_df_c['예측_판매량_v1'] = model_base.predict(future_df_c[['검침기온']])
        future_df_c['예측_판매량_v3'] = predict_piecewise_seasonal(models_final, future_df_c['검침기온'].values)
        if has_cubic_split:
            future_df_c['예측_판매량_v2'] = predict_piecewise_seasonal(models_cubic, future_df_c['검침기온'].values)
        future_df_c['Year_Month'] = future_df_c.apply(
            lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1)

        # 실제 실적이 있으면(예: 최근 진행 중인 연도) 함께 표시
        future_df_c = pd.merge(future_df_c, sales_df, on=['Year', 'Month'], how='left')
        has_actual = TARGET in future_df_c.columns and future_df_c[TARGET].notna().any()

        # 판매량 계획(기존 계획, 상품별판매량 계획 시트) 병합
        has_plan_future = False
        if plan_df is not None:
            future_df_c = pd.merge(future_df_c, plan_df, on=['Year', 'Month'], how='left')
            has_plan_future = future_df_c['판매량_계획'].notna().any()

        st.caption(f"미래 예측기온 추정: 최근 {y_years_c}개년"
                   f"({min(sim_base_years_c)}~{max(sim_base_years_c)}) 동월 실제기온 평균 사용")

        agg_cols_fut = (['판매량_계획'] if has_plan_future else []) + ([TARGET] if has_actual else []) \
            + ['예측_판매량_v1'] + (['예측_판매량_v2'] if has_cubic_split else []) + ['예측_판매량_v3']

        # 차트는 항상 전체 시리즈 표시 — 플롯리 자체 범례 클릭으로 라인 표시/숨김
        show_temp_fut = st.checkbox("🌡️ 예측기온(전월16일부터 당월15일까지) 표시", key="future_show_temp")
        render_line_chart(future_df_c, 'Year_Month', agg_cols_fut, height=420,
                          secondary_col='검침기온' if show_temp_fut else None,
                          secondary_name='예측기온(℃)')

        # 아래 선택 위젯은 표(연도별/월별)에만 반영됨 (차트에는 영향 없음)
        st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영됩니다)")
        selected_fut = st.multiselect(
            "표시할 시리즈", options=agg_cols_fut, default=agg_cols_fut,
            format_func=lambda c: SERIES_LABELS.get(c, c), key="future_series_select")
        if not selected_fut:
            st.info("표시할 항목을 1개 이상 선택해주세요. 우선 전체 항목을 표시합니다.")
            selected_fut = agg_cols_fut

        future_target_col = TARGET if has_actual else '예측_판매량_v3'
        table_series_fut = _ensure_baseline_cols(selected_fut, future_target_col, has_plan=has_plan_future)

        st.markdown("**📆 연도별 시나리오 합산**")
        yearly_future_c = render_yearly_diff_table(
            future_df_c, future_target_col, table_series_fut, key_prefix="future_yearly")

        monthly_future_diff = _build_diff_table(future_df_c, 'Year_Month', future_target_col, table_series_fut)
        monthly_future_diff = monthly_future_diff.merge(
            future_df_c[['Year_Month', '검침기온']], on='Year_Month', how='left')
        cols_order = ['Year_Month', '검침기온'] + [c for c in monthly_future_diff.columns
                                                  if c not in ('Year_Month', '검침기온')]
        disp_future = monthly_future_diff[cols_order].rename(columns={'검침기온': '예측기온'})
        st.markdown("**🗂️ 월별 시나리오**")
        render_diff_table(disp_future, 'Year_Month',
                          target_col=future_target_col if future_target_col in disp_future.columns else None,
                          key_prefix="future_monthly")

        csv_future_c = disp_future.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 냉방용 미래 시나리오 다운로드", data=csv_future_c,
                           file_name="냉방용_미래시나리오.csv", mime="text/csv")
    else:
        st.info("좌측에서 미래 시나리오 추정 연도를 선택하면 결과가 표시됩니다.")


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
            "🔍 공급량 예측 검증",
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
    # ── TAB 2: 공급량 예측 검증 ──
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[1]:
        st.markdown("### 🔍 공급량 예측 검증")
        st.markdown("""
        <div class="info-box">
        선택한 <b>학습 연도</b>로 모델을 만들고, <b>검증 연도</b>의 <u>실제 기온</u>을 넣어
        예측값을 산출한 뒤 실적과 비교합니다.<br>
        4가지 방식을 나란히 비교: <b>① Poly-3 단일</b> · <b>② 분리·3차식</b>(참고) · <b>③ 분리·2차식</b> · <b>④ 단순N년평균</b><br>
        검증 R²/MAE가 양호하면 → 아래 <b>📈 미래 예측</b> 섹션에서 바로 예측을 수행하세요.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            vf_products = st.multiselect("검증 상품 선택", options=available_products,
                default=["개별난방용"] if "개별난방용" in available_products else available_products[:1],
                key="vf_products")
        with c2:
            vf_train_years = st.multiselect("학습 연도 선택", options=years_all,
                default=years_all[-3:] if len(years_all) >= 3 else years_all,
                key="vf_train_years")

        vf_eval_years = st.multiselect(
            "🔎 검증 연도 선택 (실적이 있는 연도 — 예측 vs 실적 비교 대상)",
            options=years_all,
            default=years_all[-2:] if len(years_all) >= 2 else years_all,
            key="vf_eval_years")
        st.caption("👆 검증 연도의 실제 기온으로 예측한 뒤 실적과 비교합니다. "
                   "학습 연도와 겹쳐도 되지만, 겹치지 않을수록 진짜 예측력을 평가할 수 있습니다.")

        if not vf_products or not vf_train_years or not vf_eval_years:
            st.warning("👈 상품, 학습 연도, 검증 연도를 모두 선택해주세요.")
            st.stop()

        train_data_vf = merged[merged["연"].isin(vf_train_years)]
        eval_data_vf  = merged[merged["연"].isin(vf_eval_years)]

        if len(train_data_vf) < 12:
            st.error("학습 데이터가 12건 미만입니다. 학습 연도를 추가해주세요.")
            st.stop()
        if eval_data_vf.empty:
            st.error("검증 연도에 해당하는 데이터가 없습니다.")
            st.stop()

        x_train_vf = train_data_vf["월평균기온"].values.astype(float)

        # 단순N년평균 베이스라인용
        naive_label = f"단순{len(vf_train_years)}년평균"
        naive_by_product = {}
        for prod in vf_products:
            naive_by_product[prod] = train_data_vf.groupby("월")[prod].mean()

        # 모델 설명
        st.markdown("---")
        st.markdown(f"""
**모델 비교 설명**
- **Poly-3 단일**: 전체 기온 범위를 하나의 3차 다항식으로 학습 (기존 방식)
- **분리·3차식**: 동절기(≤{WINTER_T:.0f}℃)와 하절기(≥{SUMMER_T:.0f}℃)를 각각 3차식으로 분리 학습 (참고용)
- **분리·2차식**: 동절기/하절기를 각각 2차식으로 분리 학습 (하절기 표본 부족 시 3차식보다 안정적)
- **{naive_label}**: 학습 연도의 월별 평균값을 그대로 사용 (기온 무관 베이스라인)
""")

        for prod in vf_products:
            st.markdown(f'<div class="sub">📦 {prod}</div>', unsafe_allow_html=True)
            y_train_vf = train_data_vf[prod].values.astype(float)
            x_eval = eval_data_vf["월평균기온"].values.astype(float)
            y_actual = eval_data_vf[prod].values.astype(float)

            # ── 모델 1: Poly-3 단일 ──
            y_pred_v1, r2_train, model_vf, poly_vf = fit_poly3(x_train_vf, y_train_vf, x_eval)

            # ── 모델 2: 분리·3차식 ──
            train_for_split = train_data_vf[["월평균기온", prod]].rename(
                columns={"월평균기온": "기온_split", prod: "공급량_split"})
            models_v2, w_data_v2, s_data_v2 = fit_piecewise_seasonal_models(
                train_for_split, x_col="기온_split", y_col="공급량_split", degree=3)
            has_v2 = models_v2["winter"] is not None and models_v2["summer"] is not None
            y_pred_v2 = predict_piecewise_seasonal(models_v2, x_eval) if has_v2 else np.full_like(x_eval, np.nan)

            # ── 모델 3: 분리·2차식 ──
            models_v3, w_data_v3, s_data_v3 = fit_piecewise_seasonal_models(
                train_for_split, x_col="기온_split", y_col="공급량_split", degree=2)
            has_v3 = models_v3["winter"] is not None and models_v3["summer"] is not None
            y_pred_v3 = predict_piecewise_seasonal(models_v3, x_eval) if has_v3 else np.full_like(x_eval, np.nan)

            # ── 모델 4: 단순평균 ──
            naive_vals = eval_data_vf["월"].map(naive_by_product[prod]).values.astype(float)

            # 검증 R²/MAE 계산
            valid_mask = ~np.isnan(y_pred_v1) & ~np.isnan(y_actual)
            if valid_mask.sum() < 3:
                st.warning(f"{prod}: 검증 가능한 데이터가 3건 미만입니다.")
                continue

            def _calc_r2_mae(y_true, y_pred_arr):
                vm = ~np.isnan(y_pred_arr) & ~np.isnan(y_true)
                if vm.sum() < 3:
                    return np.nan, np.nan
                return (r2_score(y_true[vm], y_pred_arr[vm]),
                        np.mean(np.abs(y_pred_arr[vm] - y_true[vm])))

            r2_v1, mae_v1 = _calc_r2_mae(y_actual, y_pred_v1)
            r2_v2, mae_v2 = _calc_r2_mae(y_actual, y_pred_v2)
            r2_v3, mae_v3 = _calc_r2_mae(y_actual, y_pred_v3)
            r2_naive, mae_naive = _calc_r2_mae(y_actual, naive_vals)

            # R²/MAE 카드 — MAE가 가장 낮은 카드에 ✅ 표시
            metrics_vf = [
                {"label": "Poly-3 단일", "r2": r2_v1, "mae": mae_v1, "delta": None},
            ]
            if has_v2:
                metrics_vf.append({"label": "분리·3차식(참고)", "r2": r2_v2, "mae": mae_v2,
                                   "delta": r2_v2 - r2_v1 if not np.isnan(r2_v2) else None})
            if has_v3:
                metrics_vf.append({"label": "분리·2차식", "r2": r2_v3, "mae": mae_v3,
                                   "delta": r2_v3 - r2_v1 if not np.isnan(r2_v3) else None})
            metrics_vf.append({"label": f"{naive_label}", "r2": r2_naive, "mae": mae_naive, "delta": None})

            valid_maes = [m["mae"] for m in metrics_vf if not np.isnan(m["mae"])]
            best_mae = min(valid_maes) if valid_maes else None
            mcols_vf = st.columns(len(metrics_vf))
            for i, m in enumerate(metrics_vf):
                lbl = f'✅ {m["label"]}' if (best_mae is not None and m["mae"] == best_mae) else m["label"]
                if not np.isnan(m["r2"]) and not np.isnan(m["mae"]):
                    render_r2_mae_card(mcols_vf[i], lbl, m["r2"], m["mae"], delta_r2=m["delta"])
                else:
                    mcols_vf[i].markdown(f'<div style="font-size:0.8rem;color:#666;">{m["label"]}</div>'
                                         '<div style="color:#999;">데이터 부족</div>', unsafe_allow_html=True)

            st.caption(f"Poly-3 학습 R² = {r2_train:.4f} | {poly_eq_text(model_vf)}")

            # 분리 모델 수식 표시
            if has_v3:
                cw_vf = models_v3["winter"].named_steps["linearregression"].coef_
                iw_vf = models_v3["winter"].named_steps["linearregression"].intercept_
                cs_vf = models_v3["summer"].named_steps["linearregression"].coef_
                is_vf = models_v3["summer"].named_steps["linearregression"].intercept_
                r2_w_vf = r2_score(w_data_v3["공급량_split"],
                                   models_v3["winter"].predict(w_data_v3[["기온_split"]]))
                r2_s_vf = r2_score(s_data_v3["공급량_split"],
                                   models_v3["summer"].predict(s_data_v3[["기온_split"]]))
                st.caption(f"분리·2차식 — 동절기(n={len(w_data_v3)}, R²={r2_w_vf:.4f}): "
                           f"{poly_eq_str(cw_vf, iw_vf)} | "
                           f"하절기(n={len(s_data_v3)}, R²={r2_s_vf:.4f}): "
                           f"{poly_eq_str(cs_vf, is_vf)}")

            # 비교 DataFrame 구성
            eval_comp = eval_data_vf[["연", "월"]].copy()
            eval_comp["Year_Month"] = eval_comp.apply(
                lambda r: f"{int(r['연'])}-{int(r['월']):02d}", axis=1)
            eval_comp["실적"] = y_actual
            eval_comp["Poly-3 단일"] = np.round(y_pred_v1).astype(float)
            if has_v2:
                eval_comp["분리·3차식"] = np.round(y_pred_v2).astype(float)
            if has_v3:
                eval_comp["분리·2차식"] = np.round(y_pred_v3).astype(float)
            eval_comp[naive_label] = np.round(naive_vals).astype(float)
            eval_comp["월평균기온"] = x_eval

            # 차트용 컬럼 목록
            chart_cols = ["실적", "Poly-3 단일"]
            if has_v2:
                chart_cols.append("분리·3차식")
            if has_v3:
                chart_cols.append("분리·2차식")
            chart_cols.append(naive_label)

            # 라인차트
            show_temp_vf = st.checkbox("🌡️ 실제기온 표시", key=f"vf_temp_{prod}")
            render_line_chart(eval_comp, "Year_Month", chart_cols, height=420,
                              secondary_col="월평균기온" if show_temp_vf else None,
                              secondary_name="실제기온(℃)")

            # 표에 표시할 항목 선택
            st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영)")
            selected_vf = st.multiselect(
                "표시할 시리즈", options=chart_cols, default=chart_cols,
                key=f"vf_series_{prod}")
            if not selected_vf:
                st.info("표시할 항목을 1개 이상 선택해주세요.")
                selected_vf = chart_cols
            table_series_vf = _ensure_baseline_cols(selected_vf, "실적", has_plan=False)

            # 연도별 요약 (먼저 표시 — 냉방용 탭과 동일 구성)
            st.markdown("**📆 연도별 실적 대비 차이 요약**")
            yearly_table_vf = render_yearly_diff_table(eval_comp, "실적", table_series_vf,
                                     key_prefix=f"vf_yearly_{prod}",
                                     target_label="실적")

            # 월별 차이표
            diff_df = _build_diff_table(eval_comp, "Year_Month", "실적",
                                        table_series_vf, target_label="실적")
            st.markdown("**🗂️ 월별 상세 비교**")
            render_diff_table(diff_df, "Year_Month", target_col="실적",
                              key_prefix=f"vf_monthly_{prod}")

            # 산점도
            with st.expander(f"🔎 {prod} — 기온↔공급량 산점도 (학습 데이터)"):
                fig_sc = _make_scatter_chart(x_train_vf, y_train_vf,
                    f"{prod} — 기온 vs 공급량", "기온 (℃)", "공급량 (MJ)", r2_train)
                st.plotly_chart(fig_sc, use_container_width=True,
                                config=dict(scrollZoom=True, displaylogo=False))

            # CSV 다운로드
            csv_vf = diff_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(f"📥 {prod} 검증 결과 다운로드", data=csv_vf,
                               file_name=f"공급량검증_{prod}.csv", mime="text/csv",
                               key=f"dl_vf_{prod}")
            st.markdown("---")

        # ══════════════════════════════════════
        # ── 검증 탭 내 미래 예측 섹션 ──
        # ══════════════════════════════════════
        st.markdown("### 📈 미래 공급량 예측")
        st.markdown("""
        <div class="info-box">
        위 검증에서 사용한 <b>학습 연도·상품</b> 설정을 그대로 이어받아 미래 예측을 수행합니다.<br>
        아래에서 <b>예상기온 산출 기준 연도</b>와 <b>예측 기간</b>을 설정하세요.<br>
        4가지 모델 예측을 비교한 뒤, 하단에서 <b>모델을 선택</b>하면 Best/Conservative 시나리오도 확인할 수 있습니다.
        </div>
        """, unsafe_allow_html=True)

        _max_temp_year_vf = max(years_all)
        default_temp_years_vf = sorted([y for y in (_max_temp_year_vf, _max_temp_year_vf - 1, _max_temp_year_vf - 2)
                                         if y in years_all])
        if not default_temp_years_vf:
            default_temp_years_vf = years_all[-3:] if len(years_all) >= 3 else years_all

        temp_avg_years_vf = st.multiselect(
            "🌡️ 과거기온 선택 (예상기온 산출 기준 연도 · 기본값: 최근 3년 평균)",
            options=years_all,
            default=default_temp_years_vf,
            key="temp_avg_years_vf")
        st.caption("👆 미래(예측 기간)의 '예상기온'을 계산할 때 평균낼 연도. "
                   "학습 연도와 별개로 원하는 연도만 골라 월별 평균기온을 낼 수 있습니다.")

        st.markdown('<div class="sub">📅 예측 기간</div>', unsafe_allow_html=True)
        pc1_vf, pc2_vf, pc3_vf, pc4_vf = st.columns(4)
        with pc1_vf:
            pred_start_y_vf = st.selectbox("시작 연도", list(range(2020, 2036)), index=6, key="pred_sy_vf")
        with pc2_vf:
            pred_start_m_vf = st.selectbox("시작 월", list(range(1, 13)), index=0, key="pred_sm_vf")
        with pc3_vf:
            pred_end_y_vf = st.selectbox("종료 연도", list(range(2020, 2036)), index=7, key="pred_ey_vf")
        with pc4_vf:
            pred_end_m_vf = st.selectbox("종료 월", list(range(1, 13)), index=11, key="pred_em_vf")

        if st.button("🧮 공급량 예측 실행", type="primary", key="btn_supply_pred_vf"):
            st.session_state["supply_pred_run"] = True

        if st.session_state.get("supply_pred_run", False):
            if not vf_products:
                st.warning("예측할 상품을 선택해주세요 (상단 '검증 상품 선택')."); st.stop()
            if not vf_train_years:
                st.warning("학습 연도를 선택해주세요."); st.stop()
            if not temp_avg_years_vf:
                st.warning("과거기온(예상기온 산출 기준) 연도를 선택해주세요."); st.stop()

            train_data_pred = merged[merged["연"].isin(vf_train_years)]
            if len(train_data_pred) < 12:
                st.error("학습 데이터가 12건 미만입니다. 학습 연도를 추가해주세요."); st.stop()

            temp_basis_pred = merged[merged["연"].isin(temp_avg_years_vf)]
            if temp_basis_pred.empty:
                st.error("과거기온 연도에 해당하는 데이터가 없습니다."); st.stop()

            x_train_pred = train_data_pred["월평균기온"].values.astype(float)

            f_start_vf = pd.Timestamp(year=pred_start_y_vf, month=pred_start_m_vf, day=1)
            f_end_vf   = pd.Timestamp(year=pred_end_y_vf,   month=pred_end_m_vf,   day=1)
            if f_end_vf < f_start_vf:
                st.error("예측 종료가 시작보다 앞입니다."); st.stop()

            fut_months_vf = pd.date_range(start=f_start_vf, end=f_end_vf, freq="MS")
            fut_df_vf = pd.DataFrame({"연": fut_months_vf.year, "월": fut_months_vf.month})

            # 예상기온 산출 (Normal = Δ0℃ 기준)
            monthly_avg_vf = temp_basis_pred.groupby("월")["월평균기온"].mean()
            if forecast_temp_df is not None:
                fut_df_vf = fut_df_vf.merge(forecast_temp_df[["연", "월", "예상기온"]],
                    on=["연", "월"], how="left")
                miss_vf = fut_df_vf["예상기온"].isna()
                if miss_vf.any():
                    fut_df_vf.loc[miss_vf, "예상기온"] = fut_df_vf.loc[miss_vf, "월"].map(monthly_avg_vf)
            else:
                fut_df_vf["예상기온"] = fut_df_vf["월"].map(monthly_avg_vf)

            if fut_df_vf["예상기온"].isna().any():
                st.warning("일부 월은 선택한 연도만으로는 예상기온을 정하지 못해, 전체 연도 평균으로 대신 채웠습니다.")
                overall_avg_vf = merged.groupby("월")["월평균기온"].mean()
                miss_vf2 = fut_df_vf["예상기온"].isna()
                fut_df_vf.loc[miss_vf2, "예상기온"] = fut_df_vf.loc[miss_vf2, "월"].map(overall_avg_vf)
            if fut_df_vf["예상기온"].isna().any():
                fallback_single_vf = merged["월평균기온"].mean()
                fut_df_vf["예상기온"] = fut_df_vf["예상기온"].fillna(fallback_single_vf)

            st.caption(f"🌡️ 예상기온 산출 기준: {', '.join(str(y) for y in sorted(temp_avg_years_vf))}년 월별 평균")

            # 단순N년평균 라벨
            naive_label_pred = f"단순{len(temp_avg_years_vf)}년평균"

            fut_df_vf["Year_Month"] = fut_df_vf.apply(
                lambda r: f"{int(r['연'])}-{int(r['월']):02d}", axis=1)

            for prod in vf_products:
                y_train_pred = train_data_pred[prod].values.astype(float)
                st.markdown(f'<div class="sub">📦 {prod}</div>', unsafe_allow_html=True)

                x_fut_normal = fut_df_vf["예상기온"].values.astype(float)

                # ── 모델 1: Poly-3 단일 ──
                y_p1, r2_tr_p, model_p, poly_p = fit_poly3(x_train_pred, y_train_pred, x_fut_normal)
                y_p1 = np.clip(np.rint(y_p1).astype(np.int64), 0, None)

                # ── 모델 2: 분리·3차식 ──
                train_for_split_p = train_data_pred[["월평균기온", prod]].rename(
                    columns={"월평균기온": "기온_split", prod: "공급량_split"})
                models_p2, _, _ = fit_piecewise_seasonal_models(
                    train_for_split_p, x_col="기온_split", y_col="공급량_split", degree=3)
                has_p2 = models_p2["winter"] is not None and models_p2["summer"] is not None
                y_p2 = np.clip(np.rint(predict_piecewise_seasonal(models_p2, x_fut_normal)).astype(np.int64), 0, None) \
                    if has_p2 else np.full(len(x_fut_normal), np.nan)

                # ── 모델 3: 분리·2차식 ──
                models_p3, _, _ = fit_piecewise_seasonal_models(
                    train_for_split_p, x_col="기온_split", y_col="공급량_split", degree=2)
                has_p3 = models_p3["winter"] is not None and models_p3["summer"] is not None
                y_p3 = np.clip(np.rint(predict_piecewise_seasonal(models_p3, x_fut_normal)).astype(np.int64), 0, None) \
                    if has_p3 else np.full(len(x_fut_normal), np.nan)

                # ── 모델 4: 단순N년평균 ──
                naive_monthly_prod = temp_basis_pred.groupby("월")[prod].mean()
                y_p4 = fut_df_vf["월"].map(naive_monthly_prod).values.astype(float)

                # 예측 DataFrame 구성 (냉방용 구조)
                pred_comp = fut_df_vf[["연", "월", "Year_Month", "예상기온"]].copy()
                pred_comp["Poly-3 단일"] = y_p1
                if has_p2:
                    pred_comp["분리·3차식"] = y_p2
                if has_p3:
                    pred_comp["분리·2차식"] = y_p3
                pred_comp[naive_label_pred] = np.round(y_p4).astype(float)

                # 실제 실적이 있으면(예: 진행 중인 연도) 함께 표시
                # supply_df에서 직접 가져옴 (merged는 기온 merge 필수이므로 기온 없는 달이 빠짐)
                fut_years_set = set(int(y) for y in fut_df_vf["연"].unique())
                if prod in supply_df.columns:
                    _sup_tmp = supply_df[[prod]].copy()
                    _sup_tmp["연"] = _sup_tmp.index.year
                    _sup_tmp["월"] = _sup_tmp.index.month
                    _sup_tmp = _sup_tmp[_sup_tmp["연"].isin(fut_years_set)]
                    _sup_tmp = _sup_tmp[_sup_tmp[prod] > 0]  # 0인 행 제외 (데이터 없음)
                    actual_in_fut = _sup_tmp[["연", "월", prod]].rename(columns={prod: "실적"})
                else:
                    actual_in_fut = merged[merged["연"].isin(fut_years_set)][["연", "월", prod]].rename(
                        columns={prod: "실적"})
                if not actual_in_fut.empty:
                    pred_comp = pred_comp.merge(actual_in_fut, on=["연", "월"], how="left")
                has_actual_pred = "실적" in pred_comp.columns and pred_comp["실적"].notna().any()

                # 차트용 컬럼
                agg_cols_pred = (["실적"] if has_actual_pred else []) + ["Poly-3 단일"]
                if has_p2:
                    agg_cols_pred.append("분리·3차식")
                if has_p3:
                    agg_cols_pred.append("분리·2차식")
                agg_cols_pred.append(naive_label_pred)

                st.caption(f"Poly-3 Train R² = {r2_tr_p:.4f} | {poly_eq_text(model_p)}")

                # 라인차트 (냉방용과 동일)
                show_temp_pred = st.checkbox("🌡️ 예상기온 표시", key=f"pred_temp_{prod}")
                render_line_chart(pred_comp, "Year_Month", agg_cols_pred, height=420,
                                  secondary_col="예상기온" if show_temp_pred else None,
                                  secondary_name="예상기온(℃)")

                # 표에 표시할 항목 선택
                st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영됩니다)")
                selected_pred = st.multiselect(
                    "표시할 시리즈", options=agg_cols_pred, default=agg_cols_pred,
                    key=f"pred_series_{prod}")
                if not selected_pred:
                    st.info("표시할 항목을 1개 이상 선택해주세요.")
                    selected_pred = agg_cols_pred

                pred_target_col = "실적" if has_actual_pred else "Poly-3 단일"
                table_series_pred = _ensure_baseline_cols(selected_pred, pred_target_col, has_plan=False)

                # 연도별 시나리오 합산
                st.markdown("**📆 연도별 시나리오 합산**")
                render_yearly_diff_table(pred_comp, pred_target_col, table_series_pred,
                                         key_prefix=f"pred_yearly_{prod}",
                                         target_label="실적" if has_actual_pred else "Poly-3 단일")

                # 월별 시나리오
                diff_pred = _build_diff_table(pred_comp, "Year_Month", pred_target_col,
                                               table_series_pred,
                                               target_label="실적" if has_actual_pred else "Poly-3 단일")
                diff_pred = diff_pred.merge(
                    pred_comp[["Year_Month", "예상기온"]], on="Year_Month", how="left")
                cols_order_pred = ["Year_Month", "예상기온"] + [c for c in diff_pred.columns
                                                                if c not in ("Year_Month", "예상기온")]
                disp_pred = diff_pred[cols_order_pred]
                st.markdown("**🗂️ 월별 시나리오**")
                render_diff_table(disp_pred, "Year_Month",
                                  target_col=pred_target_col if pred_target_col in disp_pred.columns else None,
                                  key_prefix=f"pred_monthly_{prod}")

                # CSV 다운로드
                csv_pred = disp_pred.to_csv(index=False).encode("utf-8-sig")
                st.download_button(f"📥 {prod} 예측 결과 다운로드", data=csv_pred,
                                   file_name=f"공급량예측_{prod}.csv", mime="text/csv",
                                   key=f"dl_pred_{prod}")

                # ── 모델 선택 → Best / Conservative 시나리오 ──
                st.markdown("---")
                st.markdown(f'<div class="sub">🎯 {prod} — 모델 선택 시나리오 (Best / Conservative)</div>',
                            unsafe_allow_html=True)

                model_options_sc = ["① Poly-3 단일"]
                if has_p2:
                    model_options_sc.append("② 분리·3차식(참고)")
                if has_p3:
                    model_options_sc.append("③ 분리·2차식")
                model_options_sc.append(f"④ {naive_label_pred}")

                sc_model_sel = st.selectbox(
                    "예측 모델 선택", options=model_options_sc, key=f"sc_model_{prod}")

                sc1_vf, sc2_vf, sc3_vf = st.columns(3)
                with sc1_vf:
                    d_norm_vf = st.number_input("Normal Δ°C", value=0.0, step=0.1,
                                                format="%.1f", key=f"d_norm_{prod}")
                with sc2_vf:
                    d_best_vf = st.number_input("Best Δ°C", value=-1.0, step=0.1,
                                                format="%.1f", key=f"d_best_{prod}")
                with sc3_vf:
                    d_cons_vf = st.number_input("Conservative Δ°C", value=1.0, step=0.1,
                                                format="%.1f", key=f"d_cons_{prod}")

                scenarios_vf = {"Normal": d_norm_vf, "Best": d_best_vf, "Conservative": d_cons_vf}
                scenario_results = {}

                for sname, delta in scenarios_vf.items():
                    x_sc = (fut_df_vf["예상기온"] + delta).values.astype(float)
                    if sc_model_sel.startswith("①"):
                        y_sc, _, _, _ = fit_poly3(x_train_pred, y_train_pred, x_sc)
                        y_sc = np.clip(np.rint(y_sc).astype(np.int64), 0, None)
                    elif sc_model_sel.startswith("②"):
                        y_sc = predict_piecewise_seasonal(models_p2, x_sc)
                        y_sc = np.clip(np.rint(y_sc).astype(np.int64), 0, None)
                    elif sc_model_sel.startswith("③"):
                        y_sc = predict_piecewise_seasonal(models_p3, x_sc)
                        y_sc = np.clip(np.rint(y_sc).astype(np.int64), 0, None)
                    else:  # ④ 단순N년평균 — 기온 보정 무관
                        y_sc = np.round(y_p4).astype(np.int64)
                    scenario_results[sname] = y_sc

                sc_table = fut_df_vf[["연", "월"]].copy()
                for sname in scenarios_vf:
                    sc_table[sname] = scenario_results[sname]
                sum_row_sc = {"연": "합계", "월": ""}
                for sname in scenarios_vf:
                    sum_row_sc[sname] = int(sc_table[sname].sum())
                sc_table_full = pd.concat([sc_table, pd.DataFrame([sum_row_sc])], ignore_index=True)
                render_centered_table(sc_table_full, int_cols=list(scenarios_vf.keys()))

                st.markdown("---")

    # ══════════════════════════════════════════
    # ── TAB 3: 공급량 예측 ──
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[2]:
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

        st.caption("👆 학습 연도: Poly-3 모델(기온↔공급량 관계식)을 학습할 때 사용할 연도")
        _max_temp_year = max(years_all)
        # ★ 수정: 과거→최신 순서로 정렬 (sorted 추가)
        default_temp_years = sorted([y for y in (_max_temp_year, _max_temp_year - 1, _max_temp_year - 2)
                                     if y in years_all])
        if not default_temp_years:  # 혹시라도 y, y-1, y-2가 데이터에 하나도 없으면 안전하게 폴백
            default_temp_years = years_all[-3:] if len(years_all) >= 3 else years_all
        temp_avg_years = st.multiselect(
            "🌡️ 학습 기온 선택 (예상기온 산출 기준 연도 · 기본값: 최근 3년 평균)",
            options=years_all,
            default=default_temp_years,
            key="temp_avg_years_v2")
        st.caption("👆 미래(예측 기간)의 '예상기온'을 계산할 때 평균낼 연도. 학습 연도와 별개로 원하는 연도만 골라 "
                   "월별 평균기온을 낼 수 있습니다 (예: 최근 3년, 5년, 혹은 특정 연도들만).")

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
            if not temp_avg_years:
                st.warning("학습 기온(예상기온 산출 기준) 연도를 선택해주세요."); st.stop()
            train_data = merged[merged["연"].isin(train_years)]
            if len(train_data) < 12:
                st.error("학습 데이터가 12건 미만입니다. 학습 연도를 추가해주세요."); st.stop()
            temp_basis_data = merged[merged["연"].isin(temp_avg_years)]
            if temp_basis_data.empty:
                st.error("학습 기온 연도에 해당하는 데이터가 없습니다. 다른 연도를 선택해주세요."); st.stop()
            x_train = train_data["월평균기온"].values.astype(float)
            f_start = pd.Timestamp(year=pred_start_y, month=pred_start_m, day=1)
            f_end   = pd.Timestamp(year=pred_end_y,   month=pred_end_m,   day=1)
            if f_end < f_start:
                st.error("예측 종료가 시작보다 앞입니다."); st.stop()
            fut_months = pd.date_range(start=f_start, end=f_end, freq="MS")
            fut_df = pd.DataFrame({"연": fut_months.year, "월": fut_months.month})
            # 예상기온 산출 기준: 학습 연도가 아니라 '학습 기온 선택'에서 고른 연도들의 월별 평균
            monthly_avg = temp_basis_data.groupby("월")["월평균기온"].mean()
            if forecast_temp_df is not None:
                fut_df = fut_df.merge(forecast_temp_df[["연", "월", "예상기온"]],
                    on=["연", "월"], how="left")
                miss = fut_df["예상기온"].isna()
                if miss.any():
                    fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(monthly_avg)
            else:
                fut_df["예상기온"] = fut_df["월"].map(monthly_avg)
            if fut_df["예상기온"].isna().any():
                st.warning("일부 월은 '학습 기온 선택' 연도만으로는 예상기온을 정하지 못해, "
                          "전체 연도 평균으로 대신 채웠습니다.")
                # temp_basis_data(사용자가 고른 연도)에 없는 월은, 그 연도들만으론 채울 수 없으므로
                # merged 전체(모든 연도)의 월별 평균으로 2차 폴백한다.
                overall_avg = merged.groupby("월")["월평균기온"].mean()
                miss = fut_df["예상기온"].isna()
                fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(overall_avg)
            if fut_df["예상기온"].isna().any():
                # merged 전체에도 없는 월(이론상 거의 없음)은 마지막으로 전체 평균 1개 값으로 채운다.
                st.warning("일부 월은 참고할 기온 데이터가 전혀 없어, 전체 평균기온 1개 값으로 대체했습니다.")
                fallback_single = merged["월평균기온"].mean()
                fut_df["예상기온"] = fut_df["예상기온"].fillna(fallback_single)
            st.caption(f"🌡️ 예상기온 산출 기준: {', '.join(str(y) for y in sorted(temp_avg_years))}년 월별 평균")
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

                # 단순 N년 평균 공급량("시즈널 네이브" 베이스라인) — '학습 기온 선택' 연도의
                # 실제 공급량(prod)을 월별로 그대로 평균낸 값. 기온 회귀식 없이, "최근 몇 년간
                # 그 달엔 대략 이만큼 썼다"만 반영하는 가장 단순한 비교 기준선.
                naive_label = f"단순{len(temp_avg_years)}년평균"
                naive_monthly = temp_basis_data.groupby("월")[prod].mean()
                naive_vals_by_month = {m: naive_monthly.get(m, np.nan) for m in range(1, 13)}

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
                fig.add_trace(go.Scatter(
                    x=[f"{m}월" for m in range(1, 13)],
                    y=[naive_vals_by_month[m] for m in range(1, 13)],
                    mode="lines+markers",
                    name=f"{naive_label}({min(temp_avg_years)}~{max(temp_avg_years)})",
                    line=dict(dash="dot", color="#7c3aed", width=2.5),
                    marker=dict(symbol="diamond", size=7),
                    hovertemplate="%{x} %{y:,.0f} MJ<extra></extra>"))
                fig.update_layout(**CHART_LAYOUT)
                fig.update_layout(
                    title=f"{prod} — Poly-3 예측 (Train R²={r2_train:.4f})",
                    xaxis_title="월", yaxis_title="공급량 (MJ)", yaxis_rangemode="tozero",
                    margin=dict(t=60, b=80), dragmode="pan",
                    legend=dict(orientation="h", yanchor="top", y=-0.13,
                                xanchor="center", x=0.5, font=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True,
                                config=dict(scrollZoom=True, displaylogo=False))
                st.caption(f"🟣 점선(다이아몬드)이 '{naive_label}' — 기온 회귀식 없이 최근 "
                          f"{len(temp_avg_years)}개년({', '.join(str(y) for y in sorted(temp_avg_years))}) "
                          "실적을 월별로 그대로 평균낸 참고선입니다.")
                st.markdown(f'<div class="sub">📋 {prod} — 시나리오별 월별 예측</div>',
                            unsafe_allow_html=True)
                compare_tbl = fut_df[["연", "월"]].copy()
                for sname in scenarios:
                    compare_tbl[sname] = scenario_tables[sname][prod].values
                compare_tbl[naive_label] = compare_tbl["월"].map(naive_vals_by_month)
                sum_row = {"연": "합계", "월": ""}
                for sname in scenarios:
                    sum_row[sname] = compare_tbl[sname].sum()
                sum_row[naive_label] = compare_tbl[naive_label].sum()
                compare_full = pd.concat([compare_tbl, pd.DataFrame([sum_row])], ignore_index=True)
                render_centered_table(compare_full, int_cols=list(scenarios.keys()) + [naive_label])
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
    # ── TAB 4: 판매량 예측 (냉방용) ──
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[3]:
        render_cooling_analysis()


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
