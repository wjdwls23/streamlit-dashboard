import streamlit as st

# 페이지 제목
st.title("나의 첫 번째 Streamlit 앱")

# 간단한 텍스트 출력
st.write("안녕하세요! Streamlit으로 만든 첫 번째 앱입니다.")

# 현재 시간 표시
import datetime
st.write(f"현재 시간: {datetime.datetime.now()}")

