import streamlit as st
import requests
import pandas as pd
import time
from pyvi import ViTokenizer # THƯ VIỆN TIỀN XỬ LÝ TIẾNG VIỆT 🇻🇳
import plotly.express as px  # THƯ VIỆN VẼ BIỂU ĐỒ TRỰC QUAN
import re # THƯ VIỆN XỬ LÝ CHUỖI VÀ KÝ TỰ (Regular Expression)

# =============================
# API
# =============================

API_KEY = st.secrets["HF_API_KEY"]
API_URL = "https://router.huggingface.co/hf-inference/models/wonrax/phobert-base-vietnamese-sentiment"
headers = {"Authorization": f"Bearer {API_KEY}"}

def clean_text(text):
    text = str(text) 
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = " ".join(text.split())
    return text

def analyze(text):
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return None
            
        text_segmented = ViTokenizer.tokenize(cleaned_text)
        r = requests.post(API_URL, headers=headers, json={"inputs": text_segmented})
        return r.json()
    except:
        return None


# =============================
# CẤU HÌNH TRANG & GIAO DIỆN (CSS)
# =============================
st.set_page_config(page_title="Phân Tích Cảm Xúc", page_icon="", layout="centered")

st.markdown("""
<style>
/* NỀN VÀ FONT CHỮ CHUNG */
.stApp{ background: radial-gradient(circle at top,#0f172a,#020617); color:white; }
.hero{ text-align:center; padding:40px; }
.hero h1{ font-size:55px; font-weight:800; background: linear-gradient(90deg,#22c1c3,#fdbb2d); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p{ color:#9ca3af; font-size:20px; }
.result-box{ padding:20px; border-radius:15px; text-align:center; font-size:22px; font-weight:700; margin-top: 20px; }
.positive{ background:#052e16; color:#4ade80; border: 1px solid #4ade80; }
.negative{ background:#450a0a; color:#f87171; border: 1px solid #f87171; }
.neutral{ background:#422006; color:#facc15; border: 1px solid #facc15; }

/* HACK GIAO DIỆN TABS */
div[data-baseweb="tab-list"] { gap: 15px; justify-content: center; }
button[data-baseweb="tab"] { border: 2px solid #475569 !important; border-radius: 10px !important; padding: 10px 30px !important; background-color: transparent !important; color: #9ca3af !important; transition: all 0.3s ease-in-out !important; }
button[data-baseweb="tab"]:hover { border-color: #22c1c3 !important; color: white !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(90deg, #1e293b, #0f172a) !important; border: 2px solid #22c1c3 !important; color: #22c1c3 !important; box-shadow: 0 4px 15px rgba(34, 193, 195, 0.3) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================
# PHẦN GIỚI THIỆU (HERO)
# =============================
st.markdown("""
<div class="hero">
    <h1>HỆ THỐNG PHÂN TÍCH CẢM XÚC</h1>
    <p>Phân tích bình luận trên mạng xã hội bằng Trí tuệ Nhân tạo</p>
</div>
""", unsafe_allow_html=True)

# =============================
# CÁC TAB TƯƠNG TÁC
# =============================
tab1, tab2 = st.tabs(["Phân tích Nhanh", "Phân tích File Dữ liệu"])

# -----------------------------
# PHẦN 1: NHẬP VĂN BẢN (TAB 1)
# -----------------------------
with tab1:
    st.write("Phân tích Một Bình luận")
    text = st.text_area("Nhập bình luận của bạn:", placeholder="Ví dụ: Ứng dụng xài mượt, nhiều mã giảm giá, 10 điểm!", height=120)
    
    if st.button("Phân tích bằng AI", key="btn_instant"):
        if text.strip() == "":
            st.warning("Vui lòng nhập bình luận trước khi phân tích!")
        else:
            with st.spinner("AI đang phân tích và tách từ..."):
                result = analyze(text)
                
                if result and isinstance(result, list):
                    label = result[0][0]["label"] 
                    
                    if label == "POS":
                        st.markdown(f"<div class='result-box positive'>Tích Cực</div>", unsafe_allow_html=True)
                    elif label == "NEG":
                        st.markdown(f"<div class='result-box negative'>Tiêu Cực</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-box neutral'>Bình Thường</div>", unsafe_allow_html=True)
                else:
                    st.error("Lỗi API hoặc Dữ liệu rỗng! Vui lòng thử lại sau vài giây hoặc kiểm tra kết nối.")

# -----------------------------
# -----------------------------
# -----------------------------
# PHẦN 2: TẢI LÊN CSV (TAB 2)
# -----------------------------
with tab2:
    st.write("Phân tích Tập dữ liệu (CSV)")
    file = st.file_uploader("Tải lên file định dạng CSV", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.write("**Xem trước toàn bộ dữ liệu:**")
        st.dataframe(df, use_container_width=True, height=250)    
        column = st.selectbox("Chọn cột chứa bình luận cần phân tích:", df.columns)
        max_rows = len(df)
        num_rows = st.slider("Chọn số lượng bình luận muốn phân tích:", min_value=1, max_value=max_rows, value=min(20, max_rows))
        
        if st.button("Bắt đầu Phân tích Dữ liệu", key="btn_dataset"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            data_to_analyze = df.head(num_rows)
            total_rows = len(data_to_analyze)
            
            for i, row in data_to_analyze.iterrows():
                status_text.text(f"Đang phân tích dòng {i+1} / {total_rows}...")
                text_val = str(row[column])
                
                # Gọi hàm phân tích từng câu
                result = analyze(text_val)
                star = "Lỗi / Timeout / Chuỗi rỗng"
                
                # Bắt lỗi an toàn
                if result and isinstance(result, list):
                    try:
                        label = result[0][0]["label"]
                        if label == "POS": star = "Tích cực"
                        elif label == "NEG": star = "Tiêu cực"
                        else: star = "Bình thường"
                    except:
                        pass
                elif isinstance(result, dict) and "error" in result:
                    # Nếu API báo lỗi (VD: Quá tải) thì in thẳng ra
                    star = f"Lỗi API: {result['error']}"
                    
                results.append({
                    "Bình luận gốc": text_val,
                    "Phân loại cảm xúc": star
                })
                
                progress_bar.progress((i + 1) / total_rows)
                time.sleep(1) # Chờ 1s để API không chặn IP
            
            status_text.empty()
            st.success("Phân tích hoàn tất!")
            
            result_df = pd.DataFrame(results)
            # Thiết lập height=250 giống y hệt bảng xem trước ở trên
            st.dataframe(result_df, use_container_width=True, height=250)
            
            # ==========================================
            # 2. VẼ BIỂU ĐỒ TRỰC QUAN HÓA (PLOTLY PIE CHART)
            # ==========================================
            st.markdown("Tổng quan Cảm xúc")
            
            sentiment_counts = result_df["Phân loại cảm xúc"].value_counts().reset_index()
            sentiment_counts.columns = ["Cảm xúc", "Số lượng"]
            
            # Lọc danh sách màu sắc dựa trên các nhãn thực tế xuất hiện
            color_discrete_map = {
                "Tích cực": "#4ade80",   
                "Tiêu cực": "#f87171",   
                "Bình thường": "#facc15"
            }
            # Gán màu xám cho tất cả các nhãn bắt đầu bằng chữ "Lỗi"
            for label in sentiment_counts["Cảm xúc"]:
                if label not in color_discrete_map:
                    color_discrete_map[label] = "#9ca3af"
            
            fig = px.pie(
                sentiment_counts, 
                values="Số lượng", 
                names="Cảm xúc",
                color="Cảm xúc",
                color_discrete_map=color_discrete_map,
                hole=0.4 
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
