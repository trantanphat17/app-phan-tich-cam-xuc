import streamlit as st
import requests
import pandas as pd
import time
from pyvi import ViTokenizer # THƯ VIỆN TIỀN XỬ LÝ TIẾNG VIỆT 🇻🇳

# =============================
# API
# =============================
API_KEY = "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"
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
# PAGE CONFIG & CSS
# =============================
st.set_page_config(page_title="AI Sentiment Studio", page_icon="🧠", layout="centered")

st.markdown("""
<style>
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

/* HACK GIAO DIỆN TABS */
div[data-baseweb="tab-list"] { gap: 20px; justify-content: center; }
div[data-baseweb="tab"] {
    height: 60px !important; font-size: 18px !important; font-weight: bold !important;
    background-color: #0f172a !important; border: 1px solid #1e293b !important; 
    border-radius: 12px !important; padding: 10px 30px !important; color: #9ca3af !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
}
div[data-baseweb="tab"]:hover { border-color: #38bdf8 !important; color: white !important; }
div[data-baseweb="tab"][aria-selected="true"] { background: #1e293b !important; border: 1px solid #22c1c3 !important; color: #22c1c3 !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================
# HERO SECTION
# =============================
st.markdown("""
<div class="hero">
    <h1>AI Sentiment Studio</h1>
    <p>Analyze social media comments using Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

# =============================
# INTERACTIVE TABS
# =============================
tab1, tab2 = st.tabs(["⚡ Instant Analysis", "📊 Dataset Analysis"])

# -----------------------------
# PHẦN 1: NHẬP TEXT
# -----------------------------
with tab1:
    st.write("### 💬 Analyze a Comment")
    text = st.text_area("Enter a comment:", placeholder="Ví dụ: Ứng dụng xài mượt, nhiều mã giảm giá, 10 điểm!", height=120)
    
    if st.button("Analyze with AI 🚀", key="btn_instant"):
        if text.strip() == "":
            st.warning("Please enter a comment")
        else:
            with st.spinner("AI đang phân tích và tách từ..."):
                result = analyze(text)
                
                if result and isinstance(result, list):
                    label = result[0][0]["label"] 
                    
                    if label == "POS":
                        st.markdown(f"<div class='result-box positive'>😍 Tích Cực (Positive)</div>", unsafe_allow_html=True)
                    elif label == "NEG":
                        st.markdown(f"<div class='result-box negative'>🤬 Tiêu Cực (Negative)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-box neutral'>😐 Bình Thường (Neutral)</div>", unsafe_allow_html=True)
                else:
                    st.error("Lỗi API! Vui lòng thử lại sau vài giây.")

# -----------------------------
# PHẦN 2: UPLOAD CSV
# -----------------------------
with tab2:
    st.write("### 📂 Analyze Dataset (CSV)")
    file = st.file_uploader("Upload CSV file", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.write("**📋 Data Preview:**")
        st.dataframe(df.head())
        
        column = st.selectbox("👉 Select comment column to analyze:", df.columns)
        
        if st.button("Start Dataset Analysis 🤖", key="btn_dataset"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            data_to_analyze = df.head(10) 
            total_rows = len(data_to_analyze)
            
            for i, row in data_to_analyze.iterrows():
                status_text.text(f"Analyzing row {i+1} of {total_rows}...")
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
                time.sleep(1.5)
            
            status_text.empty()
            st.success("✅ Analysis Complete!")
            
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)