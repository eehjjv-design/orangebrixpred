from pathlib import Path

import streamlit as st

try:
    import joblib
    import numpy as np
except ModuleNotFoundError:
    joblib = None
    np = None

MODEL_PATH = Path(__file__).resolve().parent / "brix_model.joblib"
DEFAULT_FEATURES = [
    "평균기온",
    "최저기온",
    "가조시간",
    "최저 초상온도",
]
FEATURE_ALIASES = {
    "최저초상온도": "최저 초상온도",
    "최저 초상온도": "최저 초상온도",
}


if joblib is None or np is None:
    st.set_page_config(page_title="제주 감귤 당도 예측", page_icon="🍊", layout="centered")
    st.error("필수 패키지가 설치되지 않았습니다. 아래 명령을 실행해 주세요.")
    st.code("python -m pip install -r requirements.txt\n# 또는\nuv sync")
    st.stop()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None

    model = joblib.load(MODEL_PATH)
    return model


def predict_brix(model, inputs):
    normalized = {FEATURE_ALIASES.get(key, key): value for key, value in inputs.items()}

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        values = np.array([normalized[name] for name in feature_names], dtype=float).reshape(1, -1)
    else:
        values = np.array([normalized[name] for name in DEFAULT_FEATURES], dtype=float).reshape(1, -1)

    prediction = model.predict(values)[0]
    return float(prediction)


st.set_page_config(page_title="제주 감귤 당도 예측", page_icon="🍊", layout="centered")

st.title("제주도 성산지역 감귤 당도 예측")
st.caption("평균기온, 최저기온, 가조시간, 최저초상온도를 입력하면 당도를 예측합니다.")

model = load_model()
if model is None:
    st.stop()

with st.form("brix_form"):
    col1, col2 = st.columns(2)

    with col1:
        avg_temp = st.number_input("평균기온 (°C)", min_value=-20.0, max_value=60.0, value=20.0, step=0.1)
        min_temp = st.number_input("최저기온 (°C)", min_value=-30.0, max_value=50.0, value=15.0, step=0.1)

    with col2:
        sunshine_hours = st.number_input("가조시간 (시간)", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
        lowest_ground_temp = st.number_input("최저초상온도 (°C)", min_value=-20.0, max_value=40.0, value=12.0, step=0.1)

    submitted = st.form_submit_button("당도 예측하기")

if submitted:
    feature_inputs = {
        "평균기온": avg_temp,
        "최저기온": min_temp,
        "가조시간": sunshine_hours,
        "최저초상온도": lowest_ground_temp,
    }

    try:
        predicted_brix = predict_brix(model, feature_inputs)
        st.success(f"예측 당도: {predicted_brix:.2f} °Brix")
        st.metric("예상 당도", f"{predicted_brix:.2f} °Brix")
    except Exception as exc:
        st.error(f"예측 중 오류가 발생했습니다: {exc}")
