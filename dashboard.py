
import pandas as pd
import numpy as np
import streamlit as st
# import openai 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide')

# ======= basic df ===========
def load_data():
    df_basic = pd.read_csv("df_open.csv")
    df_basic['시작시간'] = pd.to_datetime(df_basic['시작시간'])
    df_basic['방송시'] = df_basic['시작시간'].dt.hour
    df_basic['방송일'] = df_basic['시작시간'].dt.day
    return df_basic

df_basic = load_data()

# ====== plus df =========
# 명서님
plus_t = pd.read_csv('전체대분류_매출상위_분석.csv')
# 민주님
df_ad = pd.read_csv('광고상품ROI.csv')

# 경민님
open_df = pd.read_csv('open_df.csv', encoding='utf-8-sig')
# '가격 구간'을 원래의 순서로 카테고리형 변환
price_order = [
    '0만원대', '10만원대', '20만원대', '30만원대', '40만원대', '50만원대', 
    '60만원대', '70만원대', '80만원대', '90만원대', '100만원 이상'
]
open_df['가격 구간'] = pd.Categorical(open_df['가격 구간'], categories=price_order, ordered=True)
open_df['라벨링'].fillna('', inplace=True)

# ==================
df = pd.read_csv('clustered_broadcast.csv')


# ===================
# 로그인 여부
if 'login' not in st.session_state:
    st.session_state.login = False

# 셀러명 입력
if 'seller_name' not in st.session_state:
    st.session_state.seller_name = ''

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# 사이드바 관리
with st.sidebar:
    # 로그인 전
    if not st.session_state.login:
    # 셀러명 입력
        seller_input = st.text_input('셀러명을 입력하세요', placeholder='셀러001')
        if st.button('로그인'):
            if seller_input in df['스토어명'].to_list():
                st.session_state.login = True
                st.session_state.seller_name = seller_input
                st.session_state.page = 'Home'
                st.rerun()
            else:
                st.warning('❌셀러명을 확인해주세요❗️ 등록되지 않은 셀러입니다.')
    # 로그인 후
    if st.session_state.login:
        if st.session_state.page == 'Home': st.markdown('# 대시보드 홈')
        elif st.session_state.page == 'Basic': st.markdown('# Basic 대시보드')
        elif st.session_state.page == 'Plus': st.markdown('# Plus 대시보드')
        elif st.session_state.page == 'Pro': st.markdown('# Pro 대시보드')
        
        st.success(f"👋 **{st.session_state.seller_name}**님 환영합니다!")

        if st.button("로그아웃"):
            st.session_state.login = False
            st.session_state.seller_name = ''
            st.session_state.page = 'Home'
            st.rerun()
            
        st.divider()
        # 메뉴 버튼
        if st.button("Home"): 
            st.session_state.page = 'Home'
            st.rerun()
        if st.button("Basic"): 
            st.session_state.page = 'Basic'
            st.rerun()
        if st.button("Plus"): 
            st.session_state.page = 'Plus'
            st.rerun()
        if st.button("Pro"): 
            st.session_state.page = 'Pro'
            st.rerun()
    else:
        st.markdown("🔐 로그인 후 메뉴가 표시됩니다.")



if st.session_state.page == 'Home':
    
    st.title('대시보드 홈')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('# Basic')
        st.write('전체 방송 동향 소개')
        if st.session_state.login:
            if st.button("Basic 대시보드로 이동"): 
                st.session_state.page = 'Basic'


    with col2:
        st.markdown('# Plus')
        st.write('구체적 전략 수립')
        if st.session_state.login:
            if st.button("Plus 대시보드로 이동"): 
                st.session_state.page = 'Plus'
                
    with col3:
        st.markdown('# Pro')
        st.write('셀러 개인 맞춤형 행동 지침 제공')
        st.write('제목 추천 시스템')
        if st.session_state.login:
            if st.button("Pro 대시보드로 이동"): 
                st.session_state.page = 'Pro'


        
elif st.session_state.page == 'Basic':
    st.title('Basic 대시보드')
    st.caption('방송 실적 요약과 주요 트렌드를 간단하게 제공합니다.')
    
    years = df_basic['시작시간'].dt.year.dropna().unique()
    months = df_basic['시작시간'].dt.month.dropna().unique()

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("연도 선택", sorted(years), index=0)
    with col2:
        selected_month = st.selectbox("월 선택", sorted(months), index=0)

    df_b = df_basic[
        (df_basic['시작시간'].dt.year == selected_year) &
        (df_basic['시작시간'].dt.month == selected_month)
    ]
    
    # ===== 핵심 성과 지표 =====
    매출 = int(df_b['총 매출액(원)'].sum())
    전환율 = df_b['구매 전환율'].mean()
    조회수 = int(df_b['방송조회수'].sum())
    판매자수 = df_b['스토어명'].nunique()
    방송수 = df_b.shape[0]
    
    st.markdown("#### 📊 핵심 성과 지표")
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div style="background-color:#f7f9fc; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">매출액</div>
            <div style="font-size:28px; font-weight:bold;">{매출 / 1e8:.0f}백만</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:#f7f9fc; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">구매 전환율</div>
            <div style="font-size:28px; font-weight:bold;">{전환율:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background-color:#f7f9fc; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">방송 조회수</div>
            <div style="font-size:28px; font-weight:bold;">{조회수/1e6:.0f}백만</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background-color:#f7f9fc; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">판매자 수</div>
            <div style="font-size:28px; font-weight:bold;">{판매자수:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style="background-color:#f7f9fc; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">방송 수</div>
            <div style="font-size:28px; font-weight:bold;">{방송수:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")

    # ===== 시간대별 방송 실적 + 성별 타겟 분포 + 지표별 카테고리 =====
    col_left, col_right = st.columns([1, 1]) 

    with col_left:
        st.markdown("#### 시간대별 방송 실적")

        selected_metric = st.selectbox("지표를 선택하세요", ["총 매출액(원)", "방송조회수", "구매 전환율"])

        hourly = df_b.groupby('방송시').agg({
            '총 매출액(원)': 'sum',
            '방송조회수': 'sum',
            '구매 전환율': 'mean'
        }).reset_index()

        max_val = hourly[selected_metric].max()
        min_val = hourly[selected_metric].min()
            
        def label_func(val):
            if val == max_val or val == min_val:
                return f"{val:,.0f}"
            else:
                return ""

        hourly['라벨'] = hourly[selected_metric].apply(label_func)
        
        fig = px.bar(
            hourly,
            x='방송시',
            y=selected_metric,
            text='라벨'
        )
        fig.update_traces(textposition='outside')

        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("#### 성별 타겟 분포")
        gender_target_counts = df_b['성별 타겟군'].dropna().str.strip().value_counts().reset_index()
        gender_target_counts.columns = ['성별 타겟군', '건수']

        fig_donut = px.pie(
            gender_target_counts,
            names='성별 타겟군',
            values='건수',
            hole=0.5
        )
        fig_donut.update_traces(textinfo='percent+label')
        
        fig_donut.update_layout(
        height=250, 
        margin=dict(t=10, b=10, l=0, r=0),
        showlegend=True
        )

        fig_donut.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig_donut, use_container_width=True)
            
        # 지표별 상위 카테고리 표
        st.markdown("#### 지표별 상위 카테고리")
        top_sales = df_b.groupby('대분류')['총 매출액(원)'].sum().sort_values(ascending=False).head(3).index.tolist()
        top_volume = df_b.groupby('대분류')['총 판매량'].sum().sort_values(ascending=False).head(3).index.tolist()
        top_views = df_b.groupby('대분류')['방송조회수'].sum().sort_values(ascending=False).head(3).index.tolist()
        
        summary_df = pd.DataFrame({
            "총 매출액 상위": top_sales,
            "총 판매량 상위": top_volume,
            "방송조회수 상위": top_views
        })
        
        st.dataframe(summary_df, use_container_width=True)

    # ===== 매출/판매 통계 + 구매 전환율 상위 카테고리 =====
    col_left, col_right = st.columns([1, 1]) 
    
    # 매출/판매 통계
    with col_left:
        st.markdown("#### 매출/판매 통계")

        by_day = df_b.groupby('방송일').agg({
            '총 매출액(원)': 'sum',
            '총 판매량': 'sum'
        }).reset_index()

        selected_metric = st.selectbox("지표를 선택하세요", ['총 매출액(원)', '총 판매량'], key='day_metric')
        metric_label = '매출금액' if selected_metric == '총 매출액(원)' else '판매수량'

        chart_df = by_day.rename(columns={selected_metric: metric_label})

        max_val = chart_df[metric_label].max()
        min_val = chart_df[metric_label].min()

        def label_func(val):
            if val == max_val or val == min_val:
                return f"{val:,.0f}"
            else:
                return ""

        chart_df['라벨'] = chart_df[metric_label].apply(label_func)

        fig = px.line(
            chart_df,
            x='방송일',
            y=metric_label,
            text='라벨',
            markers=True
        )
        fig.update_traces(textposition="top center")
        
        fig.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            height=250
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # 구매 전환율 상위 카테고리
    with col_right:
        st.markdown("#### 구매 전환율 상위 카테고리")

        conv = (
            df_b.groupby('대분류')['구매 전환율']
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        fig = px.bar(
            conv,
            x='대분류',
            y='구매 전환율',
            title=' ',
            text_auto='.2f'
        )
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

      
    
elif st.session_state.page == 'Plus':
    st.title('Plus 대시보드')
    st.caption('전략 수립에 필요한 실전형 분석 도구로, 성과를 높일 수 있는 방향을 제시합니다.')
    
    # ======== [공통 필터 : 대분류 선택] ========== #
    all_main_cats = plus_t['대분류'].dropna().unique()
    selected_main_cat = st.selectbox("분석할 카테고리를 선택하세요", sorted(all_main_cats))

    p_t = plus_t[plus_t['대분류'] == selected_main_cat]
    df_ad_cat = df_ad[df_ad['대분류'] == selected_main_cat]
    df_price = open_df[open_df['대분류'] == selected_main_cat]
    
    # ======== 시간대 분석 ========== #
    st.markdown("#### 최적의 방송 시간대 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 기획 방송 집중 시간대")
        st.caption("✅ 이 시간대는 피해서 방송을 편성하면, 비교적 안정적인 유입을 확보할 수 있습니다.")
        
        df_planned = p_t[p_t['유형'] == '기획']

        hour_counts_planned = df_planned['방송시'].value_counts().sort_index().reset_index()
        hour_counts_planned.columns = ['방송시', '기획 방송 수']
        mean_count = hour_counts_planned['기획 방송 수'].mean()

        fig1 = px.bar(
            hour_counts_planned,
            x='방송시',
            y='기획 방송 수'
        )
        
        fig1.add_hline(
            y=mean_count,
            line_dash="dot",
            line_color="red",
            annotation_text=f"평균",
            annotation_position="top left"
        )
    
        fig1.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        category_strategies = {
            "가구/인테리어": "19~20시 고성과·고경쟁. 진입 시 전략 필요",
            "도서": "10~12시 성과 높지만 기획 방송과 겹침. 전략 편성 추천",
            "디지털/가전": "10~11시·16시 기획 방송 집중 / 13·17시 성과 우수",
            "생활/건강": "10~12시 기획 방송 몰림 / 19~21시 성과 집중",
            "스포츠/레저": "18~19시 기획 방송 집중 / 14시 단독 성과 높음",
            "식품": "10시 성과 우수하나 경쟁 심함. 콘텐츠 경쟁력 필요",
            "여가/생활편의": "11시 성과 우수·기획 적음 → 기회 시간대",
            "출산/육아": "10시 성과·기획 모두 집중. 차별화 필요",
            "패션의류": "19~20시 성과 높고 기획 방송 적음 → 기회 시간대",
            "패션잡화": "18시 성과 집중. 기획 방송과 겹침 유의",
            "화장품/미용": "17시 성과 우수. 기획 방송과 병행 전략 필요"
        }
        
        if selected_main_cat in category_strategies:
            st.caption("#### 💡 방송 전략 요약")
            st.caption(category_strategies[selected_main_cat])

    with col2:
        st.markdown("##### 성과가 높았던 시간대")
        st.caption("✅ 상위 30% 셀러는 이 시간대를 공략했습니다. 해당 시간대를 중심으로 테스트해 보세요.")

        df_open_top = p_t[(p_t['유형'] == '오픈') & (p_t['매출상위'] == '상위30%')]
                
        hour_sales_open_top = df_open_top.groupby('방송시')['총 매출액(원)'].mean().reset_index()
        
        top_hour = hour_sales_open_top.sort_values(by='총 매출액(원)', ascending=False).iloc[0]['방송시']

        hour_sales_open_top['강조'] = hour_sales_open_top['방송시'].apply(
            lambda x: '강조' if x == top_hour else '기본'
        )

        fig2 = px.bar(
            hour_sales_open_top,
            x='방송시',
            y='총 매출액(원)',
            color='강조',
            color_discrete_map={
                '강조': '#ff7f0e', 
                '기본': '#1f77b4'  
            }
        )
        
        fig2.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        
    st.markdown("---")
    
    # ======= 광고 분석 ==========
    st.markdown("#### 📢 광고 성과 분석")
    gender_groups = df_ad_cat['성별 타겟군'].dropna().unique()
    cols = st.columns(len(gender_groups))

    for i, gender in enumerate(gender_groups):
        with cols[i]:
            st.markdown(f"**{gender}**")
            subset = df_ad_cat[df_ad_cat['성별 타겟군'] == gender].sort_values(by='광고상품 ROI', ascending=False).head(5)

            fig = px.bar(
                subset,
                x='광고상품 ROI',
                y='광고상품 리스트',
                orientation='h',
                text='광고상품 ROI'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            
            if len(subset) == 1:
                fig.update_layout(height=100)
            elif len(subset) == 2:
                fig.update_layout(height=200)
            else:
                fig.update_layout(height=300)
            
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    
    # ===== 레이더 차트 ====
    color_map = {
        1: 'black',
        2: 'skyblue',
        3: 'lightgreen',
        4: 'green'
    }
    def get_emoji(label):
        if label == 'opportunity':
            return '💡'
        elif label == 'test':
            return '✅'
        else:
            return ''

    agg = (
        df_price
        .groupby('가격 구간', as_index=False)
        .agg(
            효과크기=('효과 크기', 'mean'),
            평균판매량=('1회 방송당 판매량', 'mean'),
            라벨링=('라벨링', 'first')
        )
        .dropna()
        .sort_values('가격 구간')
    )

    agg['버블크기'] = agg['효과크기'] ** 3 * 30
    agg['버블색'] = agg['효과크기'].round().map(color_map)
    agg['hover'] = "라벨링: " + agg['라벨링'].astype(str)
    agg['라벨텍스트'] = agg['라벨링'].apply(get_emoji)

    fig = px.scatter(
        agg,
        x='가격 구간',
        y='평균판매량',
        size='버블크기',
        color='효과크기',
        text = '라벨텍스트',
        color_continuous_scale=['black', 'skyblue', 'lightgreen', 'green'],
        hover_name='hover',
        title=f"{selected_main_cat} - 가격 구간별 효과 분석",
        size_max=60
    )

    fig.update_traces(
    textposition='top center',
    textfont=dict(
        size=12,
        color='black',
        family='Arial'
    ),
    marker=dict(
        line=dict(width=0)
    )
)

    fig.update_layout(
        height=450,
        margin=dict(t=40, b=40, l=30, r=30),
        xaxis_title="가격 구간",
        yaxis_title="1회 방송당 평균 판매량",
        showlegend=False,
        coloraxis_showscale=False
    )
 
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡: 기획방송에서 부진하거나 미판매하는 구간입니다. 오픈라방만의 경쟁력을 강화해보세요!")
    st.caption("✅: 타 구간 대비 높은 효과가 검증되었으나 테스트 방송이 필요한 구간입니다. 파일럿 방송을 통해 성과에 따라 추후 편성 여부를 결정하세요!")
    
    
elif st.session_state.page == 'Pro':
    st.title('Pro 대시보드')
    st.header(f'셀러명 : {st.session_state.seller_name}')
    seller_df = df[df['스토어명'] == st.session_state.seller_name].reset_index(drop=True)
    
    방송_수 = seller_df.shape[0]
    총_매출 = int(seller_df['총 매출액(원)'].mean())
    평균_조회수 = int(seller_df['방송조회수'].mean())
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            '전체 방송 수',
            f'{방송_수}개'
        )
    with col2:
        st.metric(
            '평균 조회수',
            f'{평균_조회수}회'
        )
    with col3:
        st.metric(
            '평균 매출',
            f'{총_매출:,}원'
        )
    
    st.dataframe(seller_df)