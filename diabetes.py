import streamlit as st
import pandas as pd
import joblib

# 1. 페이지 설정
st.set_page_config(
    page_title="깜찍한 당뇨 예측 AI",
    page_icon="🍭",
    layout="centered"
)

# 2. 모델 및 스케일러 로드
@st.cache_resource
def load_models():
    try:
        # Colab에서 구글 드라이브에 저장한 파일명과 연결해주세요!
        # 로컬(컴퓨터)에서 실행할 때는 app.py와 같은 폴더에 파일을 두고 아래처럼 쓰면 됩니다.
        svm_model = joblib.load("diabetes.pkl")  # 저장한 SVM 모델 파일명
        scaler = joblib.load("scaler.pkl")      # 저장한 스케일러 파일명
        return svm_model, scaler
    except:
        return None, None

svm_model, scaler = load_models()

# 3. 타이틀
st.title("🍭 당뇨 위험도 측정 슬라이더 ✨")
st.markdown("슬라이더를 좌우로 움직여서 수치를 입력해 보세요! 버튼을 누르면 예측 결과가 나옵니다. 🎈")

# 4. 입력 섹션 (기본 8개 변수 슬라이더)
st.subheader("📝 건강 수치 조절하기")

col1, col2 = st.columns(2)

with col1:
    preg = st.slider("🤰 임신 횟수", 0, 20, 1)
    glucose = st.slider("🍬 포도당 수치 (혈당)", 0, 200, 100)
    bp = st.slider("💓 혈압", 0, 150, 70)
    skin = st.slider("💪 피부 두께", 0, 100, 20)

with col2:
    insulin = st.slider("💉 인슐린", 0, 900, 80)
    bmi = st.slider("🏃 체질량지수(BMI)", 0.0, 70.0, 25.0, 0.1)
    dpf = st.slider("🧬 당뇨 가족력 지수", 0.0, 3.0, 0.5, 0.01)
    age = st.slider("🎂 나이", 1, 120, 25)

st.write("---")

# 5. 예측 및 결과 출력 버튼
if st.button("✨ 결과 확인하기! ✨", use_container_width=True):
    if svm_model is not None and scaler is not None:
        # Colab 기존 코드 형태 그대로 2차원 데이터프레임 생성 (파생변수 없음 ❌)
        input_data = pd.DataFrame(
            [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=['임신횟수', '포도당', '혈압', '피부두께', '인슐린', '체질량지수', '당뇨가족력지수', '나이']
        )
        
        # 데이터 스케일링
        input_scaled = scaler.transform(input_data)
        
        # SVM 모델로 예측 및 확률 계산
        predicted = svm_model.predict(input_scaled)
        prob = svm_model.predict_proba(input_scaled)
        diabetes_prob = prob[0][1] * 100
        
        # 결과 대시보드 출력
        st.subheader("📊 AI의 분석 결과")
        
        if predicted[0] == 1:
            st.error(f"⚠️ 예측 결과: **당뇨 위험**군에 해당합니다.")
            st.write(f"현재 데이터 기준 당뇨일 확률은 **{diabetes_prob:.1f}%**입니다. 건강 관리에 유의하세요! 🥺")
        else:
            # 정상(0)일 때만 귀엽게 풍선이 퐁퐁퐁 날아오름 🎈
            st.balloons() 
            st.success(f"🎉 예측 결과: **정상**입니다!")
            st.write(f"당뇨일 확률이 **{diabetes_prob:.1f}%**로 안전한 편입니다. 축하드려요! 야호-! 😆")
            
        # 하단에 시각적인 프로그레스 바(확률바) 표시
        st.progress(int(diabetes_prob))
        
    else:
        st.error("⚠️ 모델 파일(`diabetes`, `scaler`)을 불러올 수 없습니다. 경로를 확인해주세요!")

# 하단 풋터
st.caption("제작: 깜찍한 데이터 과학자 유진 ✨")