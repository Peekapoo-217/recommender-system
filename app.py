import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hệ Thống Gợi Ý Phụ Kiện", layout="centered", page_icon="📱")

# ==========================================
# 1. HÀM TẢI DỮ LIỆU
# ==========================================
@st.cache_data
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    svd_path = os.path.join(BASE_DIR, 'saved_model', 'svd_model.pkl')
    user_path = os.path.join(BASE_DIR, 'saved_model', 'user_profiles.pkl')
    item_path = os.path.join(BASE_DIR, 'saved_model', 'item_profiles.pkl')

    with open(svd_path, 'rb') as f:
        svd_data = pickle.load(f)
    user_profiles = pd.read_pickle(user_path)
    item_profiles = pd.read_pickle(item_path)
    
    return svd_data, user_profiles, item_profiles

try:
    svd_data, user_profiles, item_profiles = load_models()
    user_enc = svd_data['user_enc']
    item_enc = svd_data['item_enc']
    user_factors = svd_data['user_factors']
    item_factors = svd_data['item_factors']
    item_matrix = item_profiles.values
    all_items = item_profiles.index.tolist()
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}")
    st.stop()

aspect_dict_vn = {
    'Screen': 'Màn hình & Hiển thị', 'Design': 'Thiết kế & Kiểu dáng',
    'Protection': 'Độ bền & Khả năng bảo vệ', 'Price_Quality': 'Giá cả & Chất lượng',
    'Power_Charging': 'Pin & Tốc độ sạc', 'Audio': 'Âm thanh'
}

def get_product_name(asin, aspect):
    if aspect == 'Screen': return f"Kính cường lực chống xước siêu cấp"
    elif aspect == 'Protection': return f"Ốp lưng chống sốc chuẩn quân đội"
    elif aspect == 'Power_Charging': return f"Cáp sạc siêu tốc bọc dù"
    elif aspect == 'Audio': return f"Tai nghe Bluetooth True Wireless"
    elif aspect == 'Design': return f"Ốp lưng thời trang họa tiết cao cấp"
    else: return f"Phụ kiện điện thoại đa năng"

def get_hybrid_recommendations(user_id, alpha=0.6, top_n=5):
    u_idx = user_enc[user_id]
    user_vector = user_factors[u_idx].reshape(1, -1)
    
    svd_scores = np.dot(user_vector, item_factors.T).flatten()
    if svd_scores.max() != svd_scores.min():
        svd_scores_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min())
    else:
        svd_scores_norm = np.zeros_like(svd_scores)
        
    u_aspect_vector = user_profiles.loc[user_id].values.reshape(1, -1)
    aspect_sim = cosine_similarity(u_aspect_vector, item_matrix).flatten()
    aspect_sim_norm = (aspect_sim + 1) / 2
    
    hybrid_score = alpha * svd_scores_norm + (1 - alpha) * aspect_sim_norm
    
    rec_df = pd.DataFrame({'ASIN': all_items, 'hybrid_score': hybrid_score})
    return rec_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

# ==========================================
# 2. CƠ CHẾ ĐĂNG NHẬP (LOGIN SYSTEM)
# ==========================================
# Lấy ra 3 ID thật từ Dataset để làm tài khoản Demo
sample_users = user_profiles.index[:3].tolist()

# Tạo database người dùng ảo (Map username với Real ReviewerID)
USER_DB = {
    "admin": {"password": "123", "real_id": sample_users[0], "name": "Quản trị viên"},
    "khachhang1": {"password": "123", "real_id": sample_users[1], "name": "Nguyễn Văn A"},
    "khachhang2": {"password": "123", "real_id": sample_users[2], "name": "Trần Thị B"}
}

# Khởi tạo Session State để lưu trạng thái đăng nhập
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['current_user'] = None

# --- MÀN HÌNH ĐĂNG NHẬP ---
if not st.session_state['logged_in']:
    st.title("🔒 Đăng Nhập Hệ Thống")
    st.write("Vui lòng đăng nhập để hệ thống cá nhân hóa trải nghiệm cho bạn.")
    
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập (Thử: admin, khachhang1, khachhang2)")
        password = st.text_input("Mật khẩu (Thử: 123)", type="password")
        submitted = st.form_submit_button("Đăng Nhập")
        
        if submitted:
            if username in USER_DB and USER_DB[username]["password"] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = USER_DB[username]
                st.rerun() # Tải lại trang web
            else:
                st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng!")

# --- MÀN HÌNH GỢI Ý (SAU KHI ĐĂNG NHẬP) ---
else:
    current_user = st.session_state['current_user']
    real_id = current_user['real_id']
    
    st.title("📱 Cửa Hàng Phụ Kiện (AI)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"👋 Xin chào, **{current_user['name']}**! (Hệ thống đã kết nối với tài khoản ID: `{real_id}`)")
    with col2:
        if st.button("Đăng xuất", type="secondary"):
            st.session_state['logged_in'] = False
            st.session_state['current_user'] = None
            st.rerun()
            
    st.divider()
    
    # AI Phân tích sở thích (Chạy ngầm với real_id)
    user_aspect_profile = user_profiles.loc[real_id]
    top_user_aspect = user_aspect_profile.idxmax()
    top_user_aspect_vn = aspect_dict_vn.get(top_user_aspect, top_user_aspect)
    
    st.info(f"🧠 **AI Phân tích:** Dựa trên lịch sử mua sắm của bạn, chúng tôi nhận thấy bạn đặc biệt ưu tiên tiêu chí **'{top_user_aspect_vn}'**.")
    
    st.markdown("### 🎯 TOP 5 SẢN PHẨM ĐỀ XUẤT DÀNH RIÊNG CHO BẠN")
    
    # Chạy hàm gợi ý với real_id
    recs = get_hybrid_recommendations(real_id, alpha=0.6, top_n=5)
    
    for rank, row in enumerate(recs.itertuples(), 1):
        asin = row.ASIN
        hybrid_score = row.hybrid_score
        
        item_aspect_profile = item_profiles.loc[asin]
        top_item_aspect = item_aspect_profile.idxmax()
        top_item_aspect_vn = aspect_dict_vn.get(top_item_aspect, top_item_aspect)
        
        product_name = get_product_name(asin, top_item_aspect)
        
        with st.container(border=True):
            st.markdown(f"#### 🏆 Top {rank}: {product_name}")
            st.caption(f"Mã SP: `{asin}` | Độ tự tin của AI: **{hybrid_score * 100:.1f}%**")
            
            if top_user_aspect == top_item_aspect:
                st.markdown(f"💡 **Tại sao đề xuất?:** Sản phẩm này được cộng đồng đánh giá cực kỳ xuất sắc về **{top_item_aspect_vn}** - hoàn toàn khớp với tiêu chí bạn đang tìm kiếm.")
            else:
                st.markdown(f"💡 **Tại sao đề xuất?:** Sản phẩm này rất nổi bật về **{top_item_aspect_vn}** và đang được nhiều khách hàng cùng gu với bạn chọn mua.")