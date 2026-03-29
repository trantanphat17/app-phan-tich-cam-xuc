import streamlit as st
import requests
import pandas as pd
import time
import concurrent.futures # THƯ VIỆN ĐỂ CHẠY SONG SONG
from pyvi import ViTokenizer 
import plotly.express as px  
import re 

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
        if not cleaned_text: return None
        text_segmented = ViTokenizer.tokenize(cleaned_text)
        r = requests.post(API_URL, headers=headers, json={"inputs": text_segmented})
        return r.json()
    except: return None

# =============================
# CẤU HÌNH TRANG & GIAO DIỆN (CSS)
# =============================
st.set_page_config(page_title="Phân Tích Cảm Xúc", page_icon="", layout="centered")

st.markdown("""
<style>
.stApp{ background: radial-gradient(circle at top,#0f172a,#020617); color:white; }
.hero{ text-align:center; padding:40px; }
.hero h1{ font-size:55px; font-weight:800; background: linear-gradient(90deg,#22c1c3,#fdbb2d); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p{ color:#9ca3af; font-size:20px; }
.result-box{ padding:20px; border-radius:15px; text-align:center; font-size:22px; font-weight:700; margin-top: 20px; }
.positive{ background:#052e16; color:#4ade80; border: 1px solid #4ade80; }
.negative{ background:#450a0a; color:#f87171; border: 1px solid #f87171; }
.neutral{ background:#422006; color:#facc15; border: 1px solid #facc15; }

div[data-baseweb="tab-list"] { gap: 15px; justify-content: center; }
button[data-baseweb="tab"] { border: 2px solid #475569 !important; border-radius: 10px !important; padding: 10px 30px !important; background-color: transparent !important; color: #9ca3af !important; transition: all 0.3s ease-in-out !important; }
button[data-baseweb="tab"]:hover { border-color: #22c1c3 !important; color: white !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(90deg, #1e293b, #0f172a) !important; border: 2px solid #22c1c3 !important; color: #22c1c3 !important; box-shadow: 0 4px 15px rgba(34, 193, 195, 0.3) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

/* FIX LỖI BẢNG ST.TABLE */
table { width: 100% !important; }
th, td { white-space: normal !important; word-wrap: break-word !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>HỆ THỐNG PHÂN TÍCH CẢM XÚC</h1><p>Phân tích bình luận trên mạng xã hội bằng Trí tuệ Nhân tạo</p></div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Phân tích Nhanh", "Phân tích File Dữ liệu"])

# -----------------------------
# PHẦN 1: NHẬP VĂN BẢN (TAB 1)
# -----------------------------
with tab1:
    st.write("Phân tích Một Bình luận")
    text = st.text_area("Nhập bình luận của bạn:", placeholder="Ví dụ: Ứng dụng xài mượt...", height=120)
    if st.button("Phân tích bằng AI", key="btn_instant"):
        if text.strip():
            with st.spinner("AI đang phân tích..."):
                result = analyze(text)
                if result and isinstance(result, list):
                    label = result[0][0]["label"] 
                    if label == "POS": st.markdown(f"<div class='result-box positive'>Tích Cực</div>", unsafe_allow_html=True)
                    elif label == "NEG": st.markdown(f"<div class='result-box negative'>Tiêu Cực</div>", unsafe_allow_html=True)
                    else: st.markdown(f"<div class='result-box neutral'>Bình Thường</div>", unsafe_allow_html=True)

# -----------------------------
# PHẦN 2: TẢI LÊN CSV (TAB 2)
# -----------------------------
with tab2:
    st.write("Phân tích Tập dữ liệu (CSV)")
    file = st.file_uploader("Tải lên file định dạng CSV", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.write("**Xem trước dữ liệu:**")
        
        # SỬA LỖI CẮT CHỮ: Dùng st.table bọc trong st.container để có thanh cuộn và tự xuống dòng
        with st.container(height=250):
            st.table(df.head(50)) # Hiển thị tối đa 50 dòng để tối ưu hiệu năng
        
        column = st.selectbox("Chọn cột chứa bình luận:", df.columns)
        num_rows = st.slider("Số lượng bình luận:", 1, len(df), min(100, len(df)))
        
        if st.button("Bắt đầu phân tích", key="btn_dataset"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            data_to_analyze = df[column].head(num_rows).astype(str).tolist()
            total_rows = len(data_to_analyze)

            def process_single_row(text_val):
                res = analyze(text_val)
                if res and isinstance(res, list) and len(res) > 0:
                    try:
                        label = res[0][0]["label"]
                        if label == "POS": return "Tích cực"
                        elif label == "NEG": return "Tiêu cực"
                    except:
                        pass
                return "Bình thường"

            processed_count = 0
            
            # XỬ LÝ ĐA LUỒNG (SONG SONG) GIÚP CHẠY NHANH
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                for text_val, sentiment in zip(data_to_analyze, executor.map(process_single_row, data_to_analyze)):
                    results.append({
                        "Bình luận gốc": text_val,
                        "Phân loại cảm xúc": sentiment
                    })
                    
                    processed_count += 1
                    status_text.text(f"Đang xử lý: {processed_count} / {total_rows} câu...")
                    progress_bar.progress(processed_count / total_rows)

            status_text.empty()
            st.success("Phân tích hoàn tất!")
            result_df = pd.DataFrame(results)
            
            # SỬA LỖI CẮT CHỮ Ở BẢNG KẾT QUẢ
            with st.container(height=400):
                st.table(result_df)
            # tai xuong    
            csv = result_df.to_csv(index=False).encode('utf-8-sig') # utf-8-sig để Excel không lỗi font
            st.download_button(
                label="Tải file kết quả (.CSV)",
                data=csv,
                file_name="ket_qua_phan_tich.csv",
                mime="text/csv",
                type="primary"
            )

            # --- Biểu đồ Pie ---
            st.markdown("### Tổng quan Cảm xúc")
            sentiment_counts = result_df["Phân loại cảm xúc"].value_counts().reset_index()
            sentiment_counts.columns = ["Cảm xúc", "Số lượng"]
            
            color_discrete_map = {"Tích cực": "#4ade80", "Tiêu cực": "#f87171", "Bình thường": "#facc15"}
            for label in sentiment_counts["Cảm xúc"]:
                if label not in color_discrete_map:
                    color_discrete_map[label] = "#9ca3af"
            
            fig = px.pie(sentiment_counts, values="Số lượng", names="Cảm xúc", color="Cảm xúc",
                         color_discrete_map=color_discrete_map, hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            
            st.plotly_chart(fig, use_container_width=True)
