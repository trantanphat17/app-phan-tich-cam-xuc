import streamlit as st
import requests
import pandas as pd
import time
from pyvi import ViTokenizer # THƯ VIỆN TIỀN XỬ LÝ TIẾNG VIỆT 🇻🇳

# =============================
# API
# =============================
API_KEY = st.secrets["HF_API_KEY"]
API_URL = "https://router.huggingface.co/hf-inference/models/wonrax/phobert-base-vietnamese-sentiment"
headers = {"Authorization": f"Bearer {API_KEY}"}

def analyze(text):
    try:
        # BƯỚC ĂN TIỀN: Tự động tách từ tiếng Việt trước khi đưa cho AI (VD: "Ứng dụng" -> "Ứng_dụng")
        text_segmented = ViTokenizer.tokenize(text)
        
        r = requests.post(API_URL, headers=headers, json={"inputs": text_segmented})
        return r.json()
    except:
        return None

# =============================
# CẤU HÌNH TRANG & GIAO DIỆN (CSS)
# =============================
st.set_page_config(page_title="Phân tích Cảm xúc AI", page_icon="🧠", layout="centered")

st.markdown("""
<style>
/* NỀN VÀ FONT CHỮ CHUNG */
.stApp{
    background: radial-gradient(circle at top,#0f172a,#020617);
    color:white;
}
.hero{
    text-align:center;
    padding:40px;
}
.hero h1{
    font-size:55px;
    font-weight:800;
    background: linear-gradient(90deg,#22c1c3,#fdbb2d);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero p{
    color:#9ca3af;
    font-size:20px;
}
.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:700;
    margin-top: 20px;
}
.positive{ background:#052e16; color:#4ade80; border: 1px solid #4ade80; }
.negative{ background:#450a0a; color:#f87171; border: 1px solid #f87171; }
.neutral{ background:#422006; color:#facc15; border: 1px solid #facc15; }

/* ========================================= */
/* HACK GIAO DIỆN TABS - THÊM KHUNG CHO 2 NÚT TAB */
/* ========================================= */
div[data-baseweb="tab-list"] { 
    gap: 15px; 
    justify-content: center; 
}

/* Đóng khung cho từng Tab */
button[data-baseweb="tab"] {
    border: 2px solid #475569 !important; 
    border-radius: 10px !important;       
    padding: 10px 30px !important;        
    background-color: transparent !important; 
    color: #9ca3af !important;            
    transition: all 0.3s ease-in-out !important; 
}

/* Hiệu ứng khi di chuột vào (Hover) */
button[data-baseweb="tab"]:hover {
    border-color: #22c1c3 !important; 
    color: white !important;
}

/* Khung khi Tab đang được chọn (Active) */
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #1e293b, #0f172a) !important; 
    border: 2px solid #22c1c3 !important; 
    color: #22c1c3 !important;            
    box-shadow: 0 4px 15px rgba(34, 193, 195, 0.3) !important; 
}

/* Ẩn đường kẻ ngang xấu xí mặc định của Streamlit ở dưới Tab */
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { 
    display: none !important; 
}
</style>
""", unsafe_allow_html=True)
# =============================
# PHẦN GIỚI THIỆU (HERO)
# =============================
st.markdown("""
<div class="hero">
    <h1>Hệ thống Phân tích Cảm xúc AI</h1>
    <p>Phân tích bình luận trên mạng xã hội bằng Trí tuệ Nhân tạo</p>
</div>
""", unsafe_allow_html=True)

# =============================
# CÁC TAB TƯƠNG TÁC
# =============================
tab1, tab2 = st.tabs(["⚡ Phân tích Nhanh", "📊 Phân tích File Dữ liệu"])

# -----------------------------
# PHẦN 1: NHẬP VĂN BẢN (TAB 1)
# -----------------------------
with tab1:
    
        st.write("### 💬 Phân tích Một Bình luận")
        text = st.text_area("Nhập bình luận của bạn:", placeholder="Ví dụ: Ứng dụng xài mượt, nhiều mã giảm giá, 10 điểm!", height=120)
        
        if st.button("Phân tích bằng AI 🚀", key="btn_instant"):
            if text.strip() == "":
                st.warning("Vui lòng nhập bình luận trước khi phân tích!")
            else:
                with st.spinner("AI đang phân tích và tách từ..."):
                    result = analyze(text)
                    
                    if result and isinstance(result, list):
                        label = result[0][0]["label"] 
                        
                        if label == "POS":
                            st.markdown(f"<div class='result-box positive'>😍 Tích Cực</div>", unsafe_allow_html=True)
                        elif label == "NEG":
                            st.markdown(f"<div class='result-box negative'>🤬 Tiêu Cực</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='result-box neutral'>😐 Bình Thường</div>", unsafe_allow_html=True)
                    else:
                        st.error("Lỗi API! Vui lòng thử lại sau vài giây hoặc kiểm tra kết nối.")

# -----------------------------
# PHẦN 2: TẢI LÊN CSV (TAB 2)
# -----------------------------
with tab2:
    
        st.write("### 📂 Phân tích Tập dữ liệu (CSV)")
        file = st.file_uploader("Tải lên file định dạng CSV", type="csv")
        
        if file:
            df = pd.read_csv(file)
            st.write("**📋 Xem trước dữ liệu:**")
            st.dataframe(df.head())
            
            column = st.selectbox("👉 Chọn cột chứa bình luận cần phân tích:", df.columns)
            
            if st.button("Bắt đầu Phân tích Dữ liệu 🤖", key="btn_dataset"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                data_to_analyze = df.head(10) 
                total_rows = len(data_to_analyze)
                
                for i, row in data_to_analyze.iterrows():
                    status_text.text(f"Đang phân tích dòng {i+1} / {total_rows}...")
                    text_val = str(row[column])
                    result = analyze(text_val)
                    
                    if result and isinstance(result, list):
                        label = result[0][0]["label"]
                        if label == "POS": star = "😍 Tích cực"
                        elif label == "NEG": star = "🤬 Tiêu cực"
                        else: star = "😐 Bình thường"
                    else:
                        star = "Lỗi / Timeout"
                        
                    results.append({
                        "Bình luận gốc": text_val,
                        "Phân loại cảm xúc": star
                    })
                    
                    progress_bar.progress((i + 1) / total_rows)
                    time.sleep(1.5) # Chờ API để tránh bị quá tải
                
                status_text.empty()
                st.success("✅ Phân tích hoàn tất!")
                
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
