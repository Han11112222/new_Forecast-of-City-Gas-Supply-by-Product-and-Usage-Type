# app.py — 도시가스 공급·판매량 예측 시스템
# Tab 1: 학습 기간 추천 (기온 학습 기간 + Poly-3 학습 기간)
# Tab 2: 공급량 예측 (Poly-3 + Normal/Best/Conservative)
# Tab 3: 판매량 예측 (냉방용, 전월16~당월15 평균기온 기준)
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="도시가스 공급·판매량 예측", page_icon="📊", layout="wide")

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
    """
    Sheet 1에서 '[new ver] 상품별 분배(MJ)' 테이블을 파싱하여
    (날짜 인덱스, 상품별 공급량 DataFrame)을 반환.
    BIO가스 행을 건너뜀.
    반환 DataFrame: index=날짜(Timestamp), columns=상품명, 값=MJ
    """
    title_idx = _find_row_containing(raw, SUPPLY_TITLE_VARIANTS)
    if title_idx is None:
        return None, "상품별 분배 테이블을 찾을 수 없습니다."

    header_idx = title_idx + 1

    # 주택용 4행
    housing_rows = [header_idx + i for i in range(1, N_HOUSING + 1)]
    subtotal1_row = header_idx + N_HOUSING + 1

    # 기타 상품: BIO가스 행 건너뛰기
    other_before = [subtotal1_row + i for i in range(1, BIO_SKIP_AFTER_N_OTHER + 1)]
    bio_row = subtotal1_row + BIO_SKIP_AFTER_N_OTHER + 1
    n_after = N_OTHER - BIO_SKIP_AFTER_N_OTHER
    other_after = [bio_row + i for i in range(1, n_after + 1)]
    other_rows = other_before + other_after

    data_rows = housing_rows + other_rows

    # 날짜 파싱
    dates = pd.to_datetime(raw.iloc[header_idx, DATA_START_COL:], errors="coerce")
    valid_cols = [i for i, d in enumerate(dates) if pd.notna(d)]
    dates_valid = dates.iloc[valid_cols]

    if len(valid_cols) == 0:
        return None, "헤더 행에서 날짜를 인식하지 못했습니다."

    # 상품별 데이터 추출
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
    """
    Sheet 1: 상품별 공급량 실적 (MJ, 월별)
    반환: DataFrame(index=날짜, columns=상품명, 값=MJ) 또는 (None, 에러메시지)
    """
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
    """
    Sheet 2: 일별 기온/공급량 실적
    반환: DataFrame(columns=[일자, 공급량_MJ, 평균기온, 최저, 최고])
    """
    try:
        resp = requests.get(SHEET2_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return None, f"Sheet 2 로드 실패: {e}"

    raw.columns = [str(c).strip() for c in raw.columns]

    # 일자 열 찾기
    date_col = None
    for c in raw.columns:
        if c in ["일자", "날짜", "date", "Date"]:
            date_col = c
            break
    if date_col is None:
        return None, "Sheet 2에서 '일자' 열을 찾을 수 없습니다."

    # 기온 열 찾기
    temp_col = None
    for c in raw.columns:
        if "평균기온" in c or "기온" in c:
            temp_col = c
            break
    if temp_col is None:
        return None, "Sheet 2에서 '평균기온' 열을 찾을 수 없습니다."

    # 공급량 열 찾기
    supply_col = None
    for c in raw.columns:
        if "공급량" in c and "MJ" in c:
            supply_col = c
            break

    df = pd.DataFrame()
    df["일자"] = pd.to_datetime(raw[date_col], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        raw[temp_col].astype(str).str.replace(",", ""), errors="coerce"
    )
    if supply_col:
        df["공급량_MJ"] = pd.to_numeric(
            raw[supply_col].astype(str).str.replace(",", ""), errors="coerce"
        )

    # 최저/최고 기온
    for label in ["최저", "최고"]:
        for c in raw.columns:
            if label in c:
                df[label] = pd.to_numeric(
                    raw[c].astype(str).str.replace(",", ""), errors="coerce"
                )
                break

    df = df.dropna(subset=["일자", "평균기온"]).sort_values("일자").reset_index(drop=True)
    df["연"] = df["일자"].dt.year
    df["월"] = df["일자"].dt.month
    df["일"] = df["일자"].dt.day
    return df, None


@st.cache_data(ttl=1800)
def load_sheet3_sales():
    """
    Sheet 3: 상품별 판매량 실적 (월별)
    반환: DataFrame(columns=[일자, 연, 월, 취사용, 개별난방용, ..., 냉방용, ...])
    """
    try:
        resp = requests.get(SHEET3_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        return None, f"Sheet 3 로드 실패: {e}"

    raw.columns = [str(c).strip() for c in raw.columns]

    # 연/월 열 찾기
    if "연" not in raw.columns and "년" in raw.columns:
        raw.rename(columns={"년": "연"}, inplace=True)

    # 숫자 변환
    for c in raw.columns:
        if c not in ["일자", "날짜", "date"]:
            raw[c] = pd.to_numeric(
                raw[c].astype(str).str.replace(",", ""), errors="coerce"
            )

    # 날짜 열 생성 (없으면 연+월로 생성)
    if "날짜" not in raw.columns and "일자" not in raw.columns:
        if "연" in raw.columns and "월" in raw.columns:
            raw["날짜"] = pd.to_datetime(
                raw["연"].astype(int).astype(str) + "-" +
                raw["월"].astype(int).astype(str) + "-01",
                errors="coerce"
            )

    # 유효 행만
    raw = raw.dropna(subset=["연", "월"]).reset_index(drop=True)
    raw["연"] = raw["연"].astype(int)
    raw["월"] = raw["월"].astype(int)

    return raw, None


def get_monthly_avg_temp(temp_daily: pd.DataFrame) -> pd.DataFrame:
    """일별 기온 → 월평균 기온 집계"""
    return (
        temp_daily.groupby(["연", "월"])["평균기온"]
        .mean().reset_index()
        .rename(columns={"평균기온": "월평균기온"})
    )


def get_cooling_period_temp(temp_daily: pd.DataFrame) -> pd.DataFrame:
    """
    냉방용 판매량 예측을 위한 검침 기간 평균기온 계산.
    당월 판매량 = 전월 16일~말일 + 당월 1일~15일의 평균기온 기준.

    반환: DataFrame(columns=[연, 월, 검침기온])
    여기서 (연, 월)은 '판매 귀속 월'이며,
    기온은 (연, 월-1, 16~말일) + (연, 월, 1~15일)의 평균.
    """
    rows = []
    for (y, m), _ in temp_daily.groupby(["연", "월"]):
        # 당월 1~15일
        cur_half = temp_daily[
            (temp_daily["연"] == y) & (temp_daily["월"] == m) & (temp_daily["일"] <= 15)
        ]["평균기온"]

        # 전월 16~말일
        if m == 1:
            py, pm = y - 1, 12
        else:
            py, pm = y, m - 1
        prev_half = temp_daily[
            (temp_daily["연"] == py) & (temp_daily["월"] == pm) & (temp_daily["일"] >= 16)
        ]["평균기온"]

        combined = pd.concat([prev_half, cur_half])
        if len(combined) >= 5:  # 최소 5일 이상 데이터 필요
            rows.append({"연": int(y), "월": int(m), "검침기온": combined.mean()})

    return pd.DataFrame(rows)


def merge_supply_and_temp(supply_df: pd.DataFrame,
                          temp_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet 1 공급량(월별 매트릭스) + Sheet 2 월평균기온 → flat 학습 데이터.
    반환: DataFrame(columns=[연, 월, 월평균기온, 취사용, 개별난방용, ...])
    """
    # supply_df: index=날짜(Timestamp), columns=상품명
    flat = supply_df.copy()
    flat["연"] = flat.index.year
    flat["월"] = flat.index.month
    flat = flat.reset_index(drop=True)

    merged = flat.merge(temp_monthly, on=["연", "월"], how="inner")
    # 0 또는 NaN 행 제거 (데이터 없는 미래 월)
    product_cols = [c for c in merged.columns if c in PRODUCT_LIST]
    merged = merged[merged[product_cols].sum(axis=1) > 0].reset_index(drop=True)
    return merged


# ══════════════════════════════════════════════
# Poly-3 모델 함수
# ══════════════════════════════════════════════

def fit_poly3(x_train, y_train, x_pred):
    """Poly-3 회귀 적합 및 예측. 반환: (y_pred, r2, model, poly)"""
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
    """Poly-3 회귀 방정식 텍스트"""
    if model is None:
        return ""
    c = model.coef_
    c1, c2, c3 = (c[0] if len(c) > 0 else 0,
                   c[1] if len(c) > 1 else 0,
                   c[2] if len(c) > 2 else 0)
    d = model.intercept_
    return f"y = {c3:+,.4f}x³ {c2:+,.4f}x² {c1:+,.4f}x {d:+,.4f}"


def recommend_train_ranges(merged_df, product, end_year=None):
    """
    [1번째 박스] Poly-3 학습 기간 추천
    - 기온: 실적연도(end_year)의 실제 월별 기온 고정
    - 변수: Poly-3 학습 기간(시작연도)만 변경
    - R²: 학습된 모델로 실적연도 공급량 예측 vs 실적연도 실제 공급량
    """
    if end_year is None:
        end_year = int(merged_df["연"].max())
    min_year = int(merged_df["연"].min())

    # 실적연도 데이터 (실제 기온 + 실제 공급량)
    actual = merged_df[merged_df["연"] == end_year][["월", "월평균기온", product]].dropna()
    if actual.empty or len(actual) < 3:
        return pd.DataFrame()

    x_actual = actual["월평균기온"].values.astype(float)   # 실적연도 실제 기온 (고정)
    y_actual = actual[product].values.astype(float)         # 실적연도 실제 공급량

    rows = []
    for sy in range(min_year, end_year):
        n_years = end_year - sy
        # 학습 데이터: sy ~ end_year-1 (실적연도 제외)
        train = merged_df[
            (merged_df["연"] >= sy) & (merged_df["연"] < end_year)
        ][["월평균기온", product]].dropna()

        if len(train) < 12:
            rows.append({
                "시작연도": sy, "종료연도": end_year - 1,
                "기간": f"{sy}~{end_year - 1}",
                "추천연도": f"최근 {n_years}년",
                "R2": np.nan,
            })
            continue

        x_tr = train["월평균기온"].values.astype(float)
        y_tr = train[product].values.astype(float)

        # 학습 후 실적연도 실제 기온으로 예측
        y_pred, _, _, _ = fit_poly3(x_tr, y_tr, x_actual)

        # R²: 예측 vs 실적
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append({
            "시작연도": sy, "종료연도": end_year - 1,
            "기간": f"{sy}~{end_year - 1}",
            "추천연도": f"최근 {n_years}년",
            "R2": float(r2) if not np.isnan(r2) else np.nan,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    return df


def recommend_temp_period(merged_df, product, temp_daily, end_year=None,
                          fixed_train_years=3):
    """
    [2번째 박스] 기온 학습 기간 추천
    - Poly-3: 최근 fixed_train_years년 학습 고정
    - 변수: 예측 기온만 과거 N년(1/2/3/5년) 월별 평균으로 변경
    - R²: 해당 평균기온으로 예측한 공급량 vs 실적연도 실제 공급량
    """
    if end_year is None:
        end_year = int(merged_df["연"].max())

    # Poly-3 학습 데이터: 최근 fixed_train_years년 고정
    train_start = end_year - fixed_train_years
    train = merged_df[
        (merged_df["연"] >= train_start) & (merged_df["연"] < end_year)
    ][["월평균기온", product]].dropna()

    if len(train) < 12:
        return pd.DataFrame()

    x_tr = train["월평균기온"].values.astype(float)
    y_tr = train[product].values.astype(float)

    # 실적연도 실제 공급량
    actual = merged_df[merged_df["연"] == end_year][["월", product]].dropna()
    if actual.empty or len(actual) < 3:
        return pd.DataFrame()
    y_actual = actual[product].values.astype(float)

    # 월별 평균기온: 연도별로 집계
    temp_by_ym = (
        temp_daily.groupby(["연", "월"])["평균기온"]
        .mean().reset_index()
    )

    periods = [1, 2, 3, 5]
    results = []

    for n_years in periods:
        past_years = list(range(end_year - n_years, end_year))
        period_str = f"{min(past_years)}~{max(past_years)}" if len(past_years) > 1 else str(past_years[0])
        rec_label = f"과거 {n_years}년평균"

        past_temps = temp_by_ym[temp_by_ym["연"].isin(past_years)]
        if past_temps.empty:
            results.append({"기간": period_str, "추천연도": rec_label, "R2": np.nan})
            continue

        avg_by_month = past_temps.groupby("월")["평균기온"].mean().reset_index()
        avg_by_month.rename(columns={"평균기온": "예측기온"}, inplace=True)

        # 실적연도와 같은 월만 매칭
        compare = actual[["월"]].merge(avg_by_month, on="월", how="inner")
        if len(compare) < 3 or len(compare) != len(y_actual):
            results.append({"기간": period_str, "추천연도": rec_label, "R2": np.nan})
            continue

        x_pred = compare["예측기온"].values.astype(float)

        # Poly-3 예측 (고정된 모델 + 변경된 기온)
        y_pred, _, _, _ = fit_poly3(x_tr, y_tr, x_pred)

        # R²: 예측 공급량 vs 실적 공급량
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        results.append({"기간": period_str, "추천연도": rec_label, "R2": float(r2)})

    return pd.DataFrame(results)


# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════

def render_centered_table(df, float_cols=None, int_cols=None, pct_cols=None, index=False):
    """HTML 중앙정렬 테이블 렌더링"""
    float_cols = float_cols or []
    int_cols = int_cols or []
    pct_cols = pct_cols or []
    show = df.copy()
    for c in float_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.2f}"
            )
    for c in int_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{int(round(x)):,}"
            )
    for c in pct_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    st.markdown(
        show.to_html(index=index, classes="centered-table"),
        unsafe_allow_html=True,
    )


def _render_highlight_table(df, headers=None, pct_cols=None):
    """
    1순위 배경 하이라이트가 적용된 HTML 테이블.
    headers: 표시할 헤더명 리스트 (None이면 df.columns 사용)
    pct_cols: 소수점 4자리로 포맷할 컬럼명 리스트
    """
    pct_cols = pct_cols or []
    cols = list(df.columns)
    if headers is None:
        headers = cols

    html_rows = ""
    for _, row in df.iterrows():
        rank = row.iloc[0]  # 첫 번째 컬럼이 추천순위
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
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════

def main():
    st.title("📊 도시가스 공급·판매량 예측 시스템")
    st.caption("대성에너지(주) 마케팅본부 · Poly-3 기온↔공급량/판매량 회귀 모델")

    # ── 데이터 로드 (백그라운드) ──
    supply_df, err1 = load_sheet1_supply()
    temp_daily, err2 = load_sheet2_temperature()
    sales_df, err3   = load_sheet3_sales()

    if err1 or err2:
        st.error("공급량 또는 기온 데이터를 불러오지 못했습니다. 구글시트 공유 설정을 확인해주세요.")
        st.stop()

    # ── 월평균기온 집계 & 공급량+기온 병합 ──
    temp_monthly = get_monthly_avg_temp(temp_daily)
    merged = merge_supply_and_temp(supply_df, temp_monthly)

    if merged.empty:
        st.warning("공급량과 기온 데이터의 겹치는 기간이 없습니다.")
        st.stop()

    available_products = [p for p in PRODUCT_LIST if p in merged.columns]
    years_all = sorted(merged["연"].unique().astype(int))

    # ── 사이드바 구성 ──
    with st.sidebar:
        # 메뉴 (최상단)
        st.markdown("### 📋 메뉴")
        menu_options = [
            "🎯 학습 기간 추천",
            "🧊 판매량 예측 (냉방용)",
            "📈 공급량 예측",
        ]
        selected_menu = st.radio(
            "분석 메뉴", options=menu_options,
            index=0, label_visibility="collapsed",
            key="main_menu",
        )

        # 예상기온 엑셀 업로드
        st.markdown("---")
        st.markdown("### 🌡️ 예상기온 업로드")
        st.caption("미래 기온 예측값 (엑셀: 날짜/연·월, 예상기온 열)")
        uploaded_temp = st.file_uploader(
            "예상기온 엑셀 (.xlsx)",
            type=["xlsx", "xls", "csv"],
            key="upload_forecast_temp",
        )
        forecast_temp_df = None
        if uploaded_temp is not None:
            forecast_temp_df = _parse_uploaded_temp(uploaded_temp)
            if forecast_temp_df is not None:
                st.success(f"✅ 예상기온 {len(forecast_temp_df)}개월 로드")

        # 데이터 로드 상태 (접이식)
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
    # 메뉴별 화면
    # ══════════════════════════════════════════

    # ══════════════════════════════════════════
    # 학습 기간 추천
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
            rec_product = st.selectbox(
                "대상 상품", options=available_products,
                index=available_products.index("개별난방용") if "개별난방용" in available_products else 0,
                key="rec_product",
            )
        with c2:
            rec_end_year = st.selectbox(
                "실적연도", options=years_all,
                index=len(years_all) - 1,
                key="rec_end_year",
            )

        if st.button("🔎 추천 구간 계산", type="primary", key="btn_rec"):
            # ── (1) Poly-3 학습 기간 추천 ──
            st.markdown('<div class="sub">📊 Poly-3 학습 기간 추천</div>',
                        unsafe_allow_html=True)
            st.caption(f"기온 고정: {rec_end_year}년 실제 월별 기온 사용 · Poly-3 학습 기간만 변경")

            rec_df = recommend_train_ranges(merged, rec_product, end_year=rec_end_year)
            rec_all = rec_df.copy()
            rec_all.insert(0, "추천순위", range(1, len(rec_all) + 1))

            # 전체 순위 표시 (시작연도/종료연도 삭제)
            _render_highlight_table(
                rec_all[["추천순위", "기간", "추천연도", "R2"]],
                headers=["추천순위", "기간", "추천연도", "R²"],
                pct_cols=["R2"],
            )

            # 그래프
            rec_plot = rec_df.sort_values("시작연도")
            fig_r = go.Figure()

            # 추천 구간 하이라이트 (상위 3개)
            top3 = rec_all.head(3)
            palette = ["rgba(255,179,71,0.2)", "rgba(118,214,165,0.2)", "rgba(120,180,255,0.2)"]
            for i, (_, row) in enumerate(top3.iterrows()):
                if i >= 3:
                    break
                fig_r.add_shape(
                    type="rect", xref="x", yref="paper",
                    x0=int(row["시작연도"]) - 0.5, x1=int(row["종료연도"]) + 0.5,
                    y0=0, y1=1,
                    line=dict(width=0), fillcolor=palette[i % len(palette)],
                )

            # Y축 범위를 데이터에 맞게 조정 (직선처럼 보이지 않도록)
            r2_vals = rec_plot["R2"].dropna().values
            y_min = max(0, float(r2_vals.min()) - 0.01)
            y_max = min(1.0, float(r2_vals.max()) + 0.005)
            if y_max - y_min < 0.02:
                y_min = max(0, y_max - 0.03)

            fig_r.add_trace(go.Scatter(
                x=rec_plot["시작연도"], y=rec_plot["R2"],
                mode="lines+markers+text",
                text=[f"{v:.4f}" if pd.notna(v) else "" for v in rec_plot["R2"]],
                textposition="top center",
                name="R² (Poly-3)",
                hovertemplate="시작연도=%{x}<br>R²=%{y:.4f}<extra></extra>",
                line=dict(color="#2c5f8a", width=2.5, shape="spline"),
                marker=dict(size=9, color="#2c5f8a",
                            line=dict(width=1.5, color="white")),
            ))
            fig_r.update_layout(**CHART_LAYOUT)
            fig_r.update_layout(
                title=f"학습 시작연도별 R² — {rec_product} (실적연도={rec_end_year})",
                xaxis_title="학습 시작연도", yaxis_title="R² (예측 vs 실적)",
                xaxis_tickmode="linear", xaxis_dtick=1,
                yaxis_range=[y_min, y_max],
                yaxis_tickformat=".4f",
                margin=dict(t=60, b=60),
            )
            st.plotly_chart(fig_r, use_container_width=True,
                            config=dict(scrollZoom=True, displaylogo=False))

            # ── (2) 기온 학습 기간 추천 ──
            st.markdown('<div class="sub">🌡️ 기온 학습 기간 추천</div>',
                        unsafe_allow_html=True)
            st.caption(f"Poly-3 고정: 최근 3년 학습 · 기온만 과거 N년 월별 평균으로 변경하여 {rec_end_year}년 실적과 비교")

            temp_rec = recommend_temp_period(merged, rec_product, temp_daily, end_year=rec_end_year)
            if not temp_rec.empty:
                temp_rec_show = temp_rec[["기간", "추천연도", "R2"]].copy()
                temp_rec_show = temp_rec_show.sort_values("R2", ascending=False, na_position="last")
                temp_rec_show.insert(0, "추천순위", range(1, len(temp_rec_show) + 1))

                _render_highlight_table(
                    temp_rec_show,
                    headers=["추천순위", "기간", "추천연도", "R²"],
                    pct_cols=["R2"],
                )

                best_temp = temp_rec_show.iloc[0]
                st.info(f"🏆 **추천 기온 기간**: {best_temp['추천연도']} ({best_temp['기간']}, R²={best_temp['R2']:.4f})")
            else:
                st.warning("기온 학습 기간 추천 계산에 필요한 데이터가 부족합니다.")

            # ── 종합 추천 ──
            if not rec_df.empty:
                best_poly = rec_df.iloc[0]
                st.success(
                    f"📌 **종합 추천**: "
                    f"Poly-3 학습 기간 **{best_poly['기간']}** "
                    f"({best_poly['추천연도']}, R²={best_poly['R2']:.4f})"
                )

    # ══════════════════════════════════════════
    # 공급량 예측
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[2]:
        st.markdown("### 📈 공급량 예측 (Poly-3)")

        # ── 설정 ──
        c1, c2 = st.columns(2)
        with c1:
            pred_products = st.multiselect(
                "예측 상품 선택", options=available_products,
                default=["개별난방용"] if "개별난방용" in available_products else available_products[:1],
                key="pred_products",
            )
        with c2:
            train_years = st.multiselect(
                "학습 연도 선택", options=years_all,
                default=years_all[-3:] if len(years_all) >= 3 else years_all,
                key="train_years",
            )

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
            pred_start_y = st.selectbox("시작 연도", list(range(2020, 2036)),
                                        index=6, key="pred_sy")
        with pc2:
            pred_start_m = st.selectbox("시작 월", list(range(1, 13)), index=0, key="pred_sm")
        with pc3:
            pred_end_y = st.selectbox("종료 연도", list(range(2020, 2036)),
                                      index=6, key="pred_ey")
        with pc4:
            pred_end_m = st.selectbox("종료 월", list(range(1, 13)), index=11, key="pred_em")

        if st.button("🧮 공급량 예측 실행", type="primary", key="btn_supply_pred"):
            if not pred_products:
                st.warning("예측할 상품을 선택해주세요."); st.stop()
            if not train_years:
                st.warning("학습 연도를 선택해주세요."); st.stop()

            # 학습 데이터
            train_data = merged[merged["연"].isin(train_years)]
            if len(train_data) < 12:
                st.error("학습 데이터가 12건 미만입니다. 학습 연도를 추가해주세요.")
                st.stop()

            x_train = train_data["월평균기온"].values.astype(float)

            # 예측 기간 구성
            f_start = pd.Timestamp(year=pred_start_y, month=pred_start_m, day=1)
            f_end   = pd.Timestamp(year=pred_end_y,   month=pred_end_m,   day=1)
            if f_end < f_start:
                st.error("예측 종료가 시작보다 앞입니다."); st.stop()

            fut_months = pd.date_range(start=f_start, end=f_end, freq="MS")
            fut_df = pd.DataFrame({"연": fut_months.year, "월": fut_months.month})

            # 예상기온 결정: 업로드 파일 > 학습기간 월평균
            monthly_avg = train_data.groupby("월")["월평균기온"].mean()

            if forecast_temp_df is not None:
                fut_df = fut_df.merge(
                    forecast_temp_df[["연", "월", "예상기온"]],
                    on=["연", "월"], how="left",
                )
                # 빈 월은 학습기간 평균으로 채움
                miss = fut_df["예상기온"].isna()
                if miss.any():
                    fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(monthly_avg)
            else:
                fut_df["예상기온"] = fut_df["월"].map(monthly_avg)

            if fut_df["예상기온"].isna().any():
                st.warning("일부 월의 예상기온을 결정하지 못했습니다. 학습기간 월평균으로 대체합니다.")
                overall_avg = train_data.groupby("월")["월평균기온"].mean()
                miss = fut_df["예상기온"].isna()
                fut_df.loc[miss, "예상기온"] = fut_df.loc[miss, "월"].map(overall_avg)

            # ── 시나리오별 예측 ──
            scenarios = {
                "Normal": d_norm,
                "Best": d_best,
                "Conservative": d_cons,
            }

            for prod in pred_products:
                y_train = train_data[prod].values.astype(float)

                st.markdown(f'<div class="sub">📦 {prod}</div>', unsafe_allow_html=True)

                # 산점도 + 회귀곡선
                _, r2_train, model, poly = fit_poly3(x_train, y_train, x_train)
                st.caption(f"Poly-3 Train R² = {r2_train:.4f} | {poly_eq_text(model)}")

                # 시나리오별 테이블 생성
                scenario_tables = {}
                for sname, delta in scenarios.items():
                    x_fut = (fut_df["예상기온"] + delta).values.astype(float)
                    y_pred, _, _, _ = fit_poly3(x_train, y_train, x_fut)
                    y_pred = np.clip(np.rint(y_pred).astype(np.int64), 0, None)
                    tbl = fut_df[["연", "월"]].copy()
                    tbl["예상기온"] = fut_df["예상기온"] + delta
                    tbl[prod] = y_pred
                    scenario_tables[sname] = tbl

                # ── 그래프: 실적 + 예측 ──
                fig = go.Figure()

                # 실적 (최근 2~3년)
                actual_temp = temp_monthly.copy()
                for y in sorted(years_all)[-3:]:
                    act = merged[merged["연"] == y][["월", prod, "월평균기온"]].sort_values("월")
                    if act.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in act["월"]],
                        y=act[prod],
                        customdata=np.round(act["월평균기온"].values, 2),
                        mode="lines+markers",
                        name=f"{y} 실적",
                        hovertemplate="%{x} %{y:,.0f} MJ<br>기온 %{customdata:.1f}℃<extra></extra>",
                    ))

                # 예측 (Normal)
                for y in sorted(fut_df["연"].unique()):
                    tbl = scenario_tables["Normal"]
                    row = tbl[tbl["연"] == y].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in row["월"]],
                        y=row[prod],
                        customdata=np.round(row["예상기온"].values, 2),
                        mode="lines",
                        name=f"예측(Normal) {y}",
                        line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,.0f} MJ<br>기온 %{customdata:.1f}℃<extra></extra>",
                    ))

                fig.update_layout(**CHART_LAYOUT)
                fig.update_layout(
                    title=f"{prod} — Poly-3 예측 (Train R²={r2_train:.4f})",
                    xaxis_title="월", yaxis_title="공급량 (MJ)",
                    yaxis_rangemode="tozero",
                    margin=dict(t=60, b=120), dragmode="pan",
                )
                st.plotly_chart(fig, use_container_width=True,
                                config=dict(scrollZoom=True, displaylogo=False))

                # ── 월별 비교 테이블 ──
                st.markdown(f'<div class="sub">📋 {prod} — 시나리오별 월별 예측</div>',
                            unsafe_allow_html=True)

                compare_tbl = fut_df[["연", "월"]].copy()
                for sname in scenarios:
                    compare_tbl[sname] = scenario_tables[sname][prod].values

                # 합계 행
                sum_row = {"연": "합계", "월": ""}
                for sname in scenarios:
                    sum_row[sname] = compare_tbl[sname].sum()
                compare_full = pd.concat(
                    [compare_tbl, pd.DataFrame([sum_row])], ignore_index=True
                )
                render_centered_table(
                    compare_full, int_cols=list(scenarios.keys())
                )

                # ── 산점도 ──
                with st.expander(f"🔎 {prod} — 기온↔공급량 산점도 (학습 데이터)"):
                    _valid = (~np.isnan(x_train)) & (~np.isnan(y_train))
                    x_v, y_v = x_train[_valid], y_train[_valid]

                    xx = np.linspace(x_v.min() - 2, x_v.max() + 2, 200)
                    yy, _, _, _ = fit_poly3(x_v, y_v, xx)

                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=x_v, y=y_v, mode="markers",
                        name="학습 샘플", marker=dict(size=6, opacity=0.6),
                        hovertemplate="기온=%{x:.1f}℃<br>공급량=%{y:,.0f}<extra></extra>",
                    ))
                    fig_sc.add_trace(go.Scatter(
                        x=xx, y=yy, mode="lines",
                        name="Poly-3 회귀", line=dict(color="#e8501a", width=2.5),
                    ))

                    # 95% 신뢰구간
                    pred_tr, _, _, _ = fit_poly3(x_v, y_v, x_v)
                    resid_std = np.std(y_v - pred_tr)
                    fig_sc.add_trace(go.Scatter(
                        x=np.concatenate([xx, xx[::-1]]),
                        y=np.concatenate([yy + 1.96 * resid_std, (yy - 1.96 * resid_std)[::-1]]),
                        fill="toself", fillcolor="rgba(232,80,26,0.12)",
                        line=dict(width=0), name="95% 신뢰구간",
                    ))
                    fig_sc.update_layout(**CHART_LAYOUT)
                    fig_sc.update_layout(
                        title=f"{prod} — 기온 vs 공급량 (R²={r2_train:.4f})",
                        xaxis_title="기온 (℃)", yaxis_title="공급량 (MJ)",
                        margin=dict(t=60, b=60),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True,
                                    config=dict(scrollZoom=True, displaylogo=False))

            # ── 엑셀 다운로드 ──
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for sname in scenarios:
                    all_prods = fut_df[["연", "월"]].copy()
                    for prod in pred_products:
                        all_prods[prod] = scenario_tables[sname][prod].values if sname in scenario_tables else 0
                    all_prods.to_excel(writer, sheet_name=sname, index=False)
            st.download_button(
                "⬇️ 예측 결과 엑셀 다운로드",
                data=buf.getvalue(),
                file_name="공급량_예측_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ══════════════════════════════════════════
    # 판매량 예측 (냉방용)
    # ══════════════════════════════════════════
    elif selected_menu == menu_options[1]:
        st.markdown("### 🧊 판매량 예측 (냉방용)")
        st.markdown("""
        <div class="info-box">
        <b>검침 기준 기온</b>: 전월 16일~말일 + 당월 1일~15일의 평균기온을 사용<br>
        <b>모델</b>: 검침기온 ↔ 냉방용 판매량 Poly-3 회귀
        </div>
        """, unsafe_allow_html=True)

        if err3:
            st.error("판매량 데이터(Sheet 3)를 불러오지 못했습니다.")
            st.stop()

        # 냉방용 열 찾기
        cooling_col = None
        for c in sales_df.columns:
            if "냉방" in str(c) or "냉난방" in str(c):
                cooling_col = c
                break

        if cooling_col is None:
            st.error("Sheet 3에서 '냉방용' 또는 '냉난방' 열을 찾을 수 없습니다.")
            st.stop()

        st.caption(f"📦 사용 열: **{cooling_col}**")

        # 검침기간 평균기온 계산
        cooling_temp = get_cooling_period_temp(temp_daily)

        if cooling_temp.empty:
            st.error("검침기간 기온을 계산할 수 없습니다. 일별 기온 데이터를 확인해주세요.")
            st.stop()

        # 판매량 + 검침기온 병합
        sales_clean = sales_df[["연", "월", cooling_col]].copy()
        sales_clean[cooling_col] = pd.to_numeric(sales_clean[cooling_col], errors="coerce")
        sales_merged = sales_clean.merge(cooling_temp, on=["연", "월"], how="inner")
        sales_merged = sales_merged.dropna(subset=[cooling_col, "검침기온"])
        sales_merged = sales_merged[sales_merged[cooling_col] > 0].reset_index(drop=True)

        if sales_merged.empty:
            st.warning("판매량과 기온 데이터의 겹치는 기간이 없습니다.")
            st.stop()

        sales_years = sorted(sales_merged["연"].unique().astype(int))
        st.caption(f"학습 가능 기간: {min(sales_years)}~{max(sales_years)}년 ({len(sales_merged)}건)")

        # ── 설정 ──
        sc1, sc2 = st.columns(2)
        with sc1:
            cooling_train_years = st.multiselect(
                "학습 연도 선택", options=sales_years,
                default=sales_years,
                key="cooling_train_years",
            )
        with sc2:
            cooling_pred_year = st.selectbox(
                "예측 연도", options=list(range(min(sales_years), 2036)),
                index=len(sales_years) - 1,
                key="cooling_pred_year",
            )

        # ── 예측 기온 입력 방식 ──
        st.markdown('<div class="sub">🌡️ 예측 기온 입력</div>', unsafe_allow_html=True)
        temp_input_mode = st.radio(
            "방식 선택",
            ["학습기간 월평균 사용", "업로드한 예상기온 사용", "직접 입력"],
            index=0, horizontal=True, key="cooling_temp_mode",
        )

        if st.button("🧮 판매량 예측 실행", type="primary", key="btn_sales_pred"):
            train_s = sales_merged[sales_merged["연"].isin(cooling_train_years)]
            if len(train_s) < 6:
                st.error("학습 데이터가 6건 미만입니다.")
                st.stop()

            x_s_train = train_s["검침기온"].values.astype(float)
            y_s_train = train_s[cooling_col].values.astype(float)

            # Poly-3 학습
            _, r2_s, model_s, poly_s = fit_poly3(x_s_train, y_s_train, x_s_train)

            st.markdown(f'<div class="sub">📊 {cooling_col} — Poly-3 모델</div>',
                        unsafe_allow_html=True)
            st.caption(f"Train R² = {r2_s:.4f} | {poly_eq_text(model_s)}")

            # 예측 기온 결정
            pred_months = list(range(1, 13))
            monthly_avg_cool = train_s.groupby("월")["검침기온"].mean()

            if temp_input_mode == "업로드한 예상기온 사용" and forecast_temp_df is not None:
                # 업로드 파일에서 예측 연도 기온 가져오기
                fc_yr = forecast_temp_df[forecast_temp_df["연"] == cooling_pred_year]
                pred_temps = []
                for m in pred_months:
                    row = fc_yr[fc_yr["월"] == m]
                    if not row.empty:
                        pred_temps.append(float(row.iloc[0]["예상기온"]))
                    else:
                        pred_temps.append(monthly_avg_cool.get(m, np.nan))
            else:
                pred_temps = [monthly_avg_cool.get(m, np.nan) for m in pred_months]

            x_pred_cool = np.array(pred_temps, dtype=float)
            valid_mask = ~np.isnan(x_pred_cool)

            if valid_mask.sum() == 0:
                st.error("예측 기온이 모두 비어있습니다.")
                st.stop()

            # 유효한 월만 예측
            x_valid = x_pred_cool[valid_mask]
            y_pred_cool, _, _, _ = fit_poly3(x_s_train, y_s_train, x_valid)
            y_pred_cool = np.clip(np.rint(y_pred_cool).astype(np.int64), 0, None)

            # 결과 테이블
            result_tbl = pd.DataFrame({
                "월": [f"{m}월" for m in pred_months],
                "검침기온": pred_temps,
                f"예측_{cooling_col}": [np.nan] * 12,
            })
            j = 0
            for i, v in enumerate(valid_mask):
                if v:
                    result_tbl.loc[i, f"예측_{cooling_col}"] = int(y_pred_cool[j])
                    j += 1

            # 실적 비교 (있으면)
            actual_yr = sales_merged[sales_merged["연"] == cooling_pred_year]
            if not actual_yr.empty:
                act_by_m = actual_yr.set_index("월")[cooling_col]
                result_tbl[f"실적_{cooling_col}"] = [
                    act_by_m.get(m, np.nan) for m in pred_months
                ]
                result_tbl["차이"] = (
                    result_tbl[f"예측_{cooling_col}"] - result_tbl[f"실적_{cooling_col}"]
                )

            # 합계 행
            sum_row = {"월": "합계", "검침기온": ""}
            for c in result_tbl.columns:
                if c not in ["월", "검침기온"]:
                    sum_row[c] = pd.to_numeric(result_tbl[c], errors="coerce").sum()
            result_full = pd.concat(
                [result_tbl, pd.DataFrame([sum_row])], ignore_index=True
            )

            int_cols_s = [c for c in result_full.columns if c not in ["월", "검침기온"]]
            render_centered_table(result_full, float_cols=["검침기온"], int_cols=int_cols_s)

            # ── 그래프: 실적 vs 예측 ──
            fig_cool = go.Figure()

            # 실적 (학습 연도)
            for y in sorted(cooling_train_years)[-3:]:
                act = train_s[train_s["연"] == y].sort_values("월")
                fig_cool.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in act["월"]],
                    y=act[cooling_col],
                    customdata=np.round(act["검침기온"].values, 1),
                    mode="lines+markers",
                    name=f"{y} 실적",
                    hovertemplate="%{x} %{y:,.0f}<br>검침기온 %{customdata:.1f}℃<extra></extra>",
                ))

            # 예측
            pred_m_labels = [f"{m}월" for i, m in enumerate(pred_months) if valid_mask[i]]
            fig_cool.add_trace(go.Scatter(
                x=pred_m_labels, y=y_pred_cool,
                customdata=np.round(x_valid, 1),
                mode="lines+markers",
                name=f"예측 {cooling_pred_year}",
                line=dict(dash="dash", color="#e8501a", width=2.5),
                marker=dict(size=8),
                hovertemplate="%{x} %{y:,.0f}<br>검침기온 %{customdata:.1f}℃<extra></extra>",
            ))

            fig_cool.update_layout(**CHART_LAYOUT)
            fig_cool.update_layout(
                title=f"{cooling_col} 판매량 — 실적 vs 예측 (R²={r2_s:.4f})",
                xaxis_title="월", yaxis_title="판매량",
                yaxis_rangemode="tozero",
                margin=dict(t=60, b=120), dragmode="pan",
            )
            st.plotly_chart(fig_cool, use_container_width=True,
                            config=dict(scrollZoom=True, displaylogo=False))

            # ── 산점도 ──
            with st.expander(f"🔎 {cooling_col} — 검침기온↔판매량 산점도"):
                xx_s = np.linspace(x_s_train.min() - 2, x_s_train.max() + 2, 200)
                yy_s, _, _, _ = fit_poly3(x_s_train, y_s_train, xx_s)

                fig_sc_s = go.Figure()
                fig_sc_s.add_trace(go.Scatter(
                    x=x_s_train, y=y_s_train, mode="markers",
                    name="학습 샘플", marker=dict(size=6, opacity=0.6),
                ))
                fig_sc_s.add_trace(go.Scatter(
                    x=xx_s, y=yy_s, mode="lines",
                    name="Poly-3 회귀", line=dict(color="#e8501a", width=2.5),
                ))
                fig_sc_s.update_layout(**CHART_LAYOUT)
                fig_sc_s.update_layout(
                    title=f"{cooling_col} — 검침기온 vs 판매량 (R²={r2_s:.4f})",
                    xaxis_title="검침기온 (℃)", yaxis_title="판매량",
                    margin=dict(t=60, b=60),
                )
                st.plotly_chart(fig_sc_s, use_container_width=True,
                                config=dict(scrollZoom=True, displaylogo=False))


def _parse_uploaded_temp(uploaded_file):
    """업로드된 예상기온 파일 파싱 → DataFrame(연, 월, 예상기온)"""
    try:
        name = getattr(uploaded_file, "name", "")
        if name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        df.columns = [str(c).strip() for c in df.columns]

        # 날짜 또는 연/월 찾기
        if "연" not in df.columns and "월" not in df.columns:
            date_col = None
            for c in df.columns:
                if c in ["날짜", "일자", "date", "Date"]:
                    date_col = c
                    break
            if date_col is None:
                date_col = df.columns[0]
            df["날짜"] = pd.to_datetime(df[date_col], errors="coerce")
            df["연"] = df["날짜"].dt.year
            df["월"] = df["날짜"].dt.month

        # 기온 열 찾기
        temp_col = None
        for c in df.columns:
            if "예상" in c or "평균기온" in c or "기온" in c or "temp" in c.lower():
                temp_col = c
                break
        if temp_col is None:
            # 숫자 컬럼 중 연/월이 아닌 첫 번째
            for c in df.columns:
                if c not in ["연", "월", "날짜", "일자"] and pd.api.types.is_numeric_dtype(df[c]):
                    temp_col = c
                    break

        if temp_col is None:
            st.sidebar.error("예상기온 파일에서 기온 열을 찾지 못했습니다.")
            return None

        result = pd.DataFrame({
            "연": df["연"].astype(int),
            "월": df["월"].astype(int),
            "예상기온": pd.to_numeric(df[temp_col], errors="coerce"),
        }).dropna()

        return result

    except Exception as e:
        st.sidebar.error(f"예상기온 파일 파싱 실패: {e}")
        return None


if __name__ == "__main__":
    main()
