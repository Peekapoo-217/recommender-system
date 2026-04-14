import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hệ Thống Gợi Ý Phụ Kiện", layout="centered", page_icon="📱")

# ==========================================
# 1. HÀM TẢI DỮ LIỆU (Đã thêm Product Metadata)
# ==========================================
@st.cache_data
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Đường dẫn tới các file pkl
    svd_path = os.path.join(BASE_DIR, 'saved_model', 'svd_model.pkl')
    user_path = os.path.join(BASE_DIR, 'saved_model', 'user_profiles.pkl')
    item_path = os.path.join(BASE_DIR, 'saved_model', 'item_profiles.pkl')
    meta_path = os.path.join(BASE_DIR, 'saved_model', 'product_info.pkl')

    with open(svd_path, 'rb') as f:
        svd_data = pickle.load(f)
    with open(meta_path, 'rb') as f:
        product_info = pickle.load(f)

    user_profiles = pd.read_pickle(user_path)
    item_profiles = pd.read_pickle(item_path)
    
    return svd_data, user_profiles, item_profiles, product_info

try:
    svd_data, user_profiles, item_profiles, product_info = load_models()
    user_enc = svd_data['user_enc']
    item_enc = svd_data['item_enc']
    user_factors = svd_data['user_factors']
    item_factors = svd_data['item_factors']
    item_matrix = item_profiles.values
    all_items = item_profiles.index.tolist()
except Exception as e:
    st.error(f"Lỗi tải mô hình hoặc dữ liệu: {e}")
    st.stop()

# Từ điển dịch khía cạnh
aspect_dict_vn = {
    'Screen': 'Màn hình & Hiển thị', 'Design': 'Thiết kế & Kiểu dáng',
    'Protection': 'Độ bền & Khả năng bảo vệ', 'Price_Quality': 'Giá cả & Chất lượng',
    'Power_Charging': 'Pin & Tốc độ sạc', 'Audio': 'Âm thanh'
}

# ==========================================
# 2. THUẬT TOÁN GỢI Ý (Hybrid)
# ==========================================
def get_hybrid_recommendations(user_id, alpha=0.2, top_n=5):
    u_idx = user_enc[user_id]
    user_vector = user_factors[u_idx].reshape(1, -1)
    
    # 1. Điểm từ SVD (Collaborative Filtering)
    svd_scores = np.dot(user_vector, item_factors.T).flatten()
    if svd_scores.max() != svd_scores.min():
        svd_scores_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min())
    else:
        svd_scores_norm = np.zeros_like(svd_scores)
        
    # 2. Điểm từ Aspect (Content-based)
    u_aspect_vector = user_profiles.loc[user_id].values.reshape(1, -1)
    aspect_sim = cosine_similarity(u_aspect_vector, item_matrix).flatten()
    aspect_sim_norm = (aspect_sim + 1) / 2
    
    # 3. Phối hợp Hybrid (Sử dụng Alpha tối ưu 0.2)
    hybrid_score = alpha * svd_scores_norm + (1 - alpha) * aspect_sim_norm
    
    rec_df = pd.DataFrame({'ASIN': all_items, 'hybrid_score': hybrid_score})
    return rec_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

# ==========================================
# 3. HỆ THỐNG ĐĂNG NHẬP
# ==========================================
sample_users = user_profiles.index[:3].tolist()
USER_DB = {
    "admin": {"password": "123", "real_id": sample_users[0], "name": "Quản trị viên"},
    "khachhang1": {"password": "123", "real_id": sample_users[1], "name": "Nguyễn Văn A"},
    "khachhang2": {"password": "123", "real_id": sample_users[2], "name": "Trần Thị B"}
}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_user'] = None

if not st.session_state['logged_in']:
    st.title("🔒 Đăng Nhập Hệ Thống")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng Nhập")
        if submitted:
            if username in USER_DB and USER_DB[username]["password"] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = USER_DB[username]
                st.rerun()
            else:
                st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng!")
else:
    # --- GIAO DIỆN CHÍNH ---
    current_user = st.session_state['current_user']
    real_id = current_user['real_id']
    
    st.title("📱 Smart Recommendation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Chào mừng, **{current_user['name']}**!")
    with col2:
        if st.button("Đăng xuất"):
            st.session_state['logged_in'] = False
            st.rerun()
            
    st.divider()
    
    # Phân tích profile người dùng
    user_aspect_profile = user_profiles.loc[real_id]
    top_user_aspect = user_aspect_profile.idxmax()
    top_user_aspect_vn = aspect_dict_vn.get(top_user_aspect, top_user_aspect)
    
    st.info(f"🧠 **AI Insight:** Bạn có xu hướng quan tâm nhiều đến khía cạnh **{top_user_aspect_vn}**.")
    
    st.markdown("### 🎯 Sản phẩm đề xuất cho bạn")
    
    # Gọi gợi ý với Alpha = 0.2 (kết quả thực nghiệm tốt nhất)
    recs = get_hybrid_recommendations(real_id, alpha=0.2, top_n=5)
    
    for rank, row in enumerate(recs.itertuples(), 1):
        asin = row.ASIN
        score = row.hybrid_score
        
        # Lấy thông tin từ metadata thật
        item_meta = product_info.get(asin, {})
        title = item_meta.get('title', f"Sản phẩm {asin}")
        img_url = item_meta.get('imUrl') # Hoặc 'image' tùy file meta của bạn
        
        # Phân tích thế mạnh sản phẩm
        item_aspect_profile = item_profiles.loc[asin]
        top_item_aspect = item_aspect_profile.idxmax()
        top_item_aspect_vn = aspect_dict_vn.get(top_item_aspect, top_item_aspect)
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 4])
            with c1:
                if img_url:
                    st.image(img_url)
                else:
                    st.write("📦")
            with c2:
                st.markdown(f"**Top {rank}: {title}**")
                st.caption(f"Mã: `{asin}` | Độ phù hợp: **{score * 100:.1f}%**")
                
                if top_user_aspect == top_item_aspect:
                    st.write(f"✅ Khớp hoàn toàn với sở thích của bạn về **{top_item_aspect_vn}**.")
                else:
                    st.write(f"🌟 Nổi bật về **{top_item_aspect_vn}**, rất đáng để trải nghiệm.")