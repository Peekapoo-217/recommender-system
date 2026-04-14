import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hệ Thống Gợi Ý Phụ Kiện", layout="wide", page_icon="📱")

# ==========================================
# 1. TẢI DỮ LIỆU & TỪ ĐIỂN
# ==========================================
@st.cache_data
def load_all_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'saved_model')
    
    with open(os.path.join(model_path, 'svd_model.pkl'), 'rb') as f:
        svd_data = pickle.load(f)
    with open(os.path.join(model_path, 'product_info.pkl'), 'rb') as f:
        product_info = pickle.load(f)
        
    user_profiles = pd.read_pickle(os.path.join(model_path, 'user_profiles.pkl'))
    item_profiles = pd.read_pickle(os.path.join(model_path, 'item_profiles.pkl'))
    
    return svd_data, user_profiles, item_profiles, product_info

try:
    svd_data, user_profiles, item_profiles, product_info = load_all_data()
    user_enc = svd_data['user_enc']
    item_factors = svd_data['item_factors']
    user_factors = svd_data['user_factors']
    item_matrix = item_profiles.values
    all_items = item_profiles.index.tolist()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu: {e}")
    st.stop()

aspect_dict_vn = {
    'Screen': 'Màn hình & Hiển thị', 'Design': 'Thiết kế & Kiểu dáng',
    'Protection': 'Độ bền & Bảo vệ', 'Price_Quality': 'Giá cả & Chất lượng',
    'Power_Charging': 'Pin & Sạc', 'Audio': 'Âm thanh'
}

# ==========================================
# 2. KHỞI TẠO DATABASE ẢO (SESSION STATE)
# ==========================================
# Đưa Database vào Session State để nó tự động cập nhật khi có người đăng ký
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {
        "admin": {"password": "123", "real_id": user_profiles.index[0], "name": "Nguyễn Văn Admin", "is_new": False},
        "khachhang1": {"password": "123", "real_id": user_profiles.index[1], "name": "Lê Minh Công", "is_new": False}
    }

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'current_user' not in st.session_state: st.session_state['current_user'] = None

# ==========================================
# 3. HAI LUỒNG THUẬT TOÁN GỢI Ý
# ==========================================
# Luồng 1: Dành cho khách cũ (Hybrid)
def get_hybrid_recommendations(user_id, alpha=0.2):
    u_idx = user_enc[user_id]
    user_vector = user_factors[u_idx].reshape(1, -1)
    
    svd_scores = np.dot(user_vector, item_factors.T).flatten()
    svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min()) if svd_scores.max() > svd_scores.min() else svd_scores
    
    u_aspect_vector = user_profiles.loc[user_id].values.reshape(1, -1)
    aspect_sim = cosine_similarity(u_aspect_vector, item_matrix).flatten()
    aspect_sim_norm = (aspect_sim + 1) / 2
    
    hybrid_score = alpha * svd_norm + (1 - alpha) * aspect_sim_norm
    rec_df = pd.DataFrame({'ASIN': all_items, 'score': hybrid_score})
    return rec_df.sort_values(by='score', ascending=False)

# Luồng 2: Dành cho khách mới đăng ký (Content-Based)
def get_content_recommendations(custom_vector):
    sims = cosine_similarity(custom_vector, item_matrix).flatten()
    sim_scores_norm = (sims + 1) / 2
    rec_df = pd.DataFrame({'ASIN': all_items, 'score': sim_scores_norm})
    return rec_df.sort_values(by='score', ascending=False)

# Hàm hiển thị giao diện UI cho sản phẩm
def render_product_cards(recs_dataframe, is_hybrid=True, user_gu=None):
    valid_count = 0
    for row in recs_dataframe.itertuples():
        if valid_count >= 5: break
        if row.ASIN in product_info:
            info = product_info[row.ASIN]
            item_strength = item_profiles.loc[row.ASIN].idxmax()
            
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.image(info['imUrl']) if info.get('imUrl') else c1.write("📦")
                with c2:
                    st.markdown(f"**{info.get('title', 'Sản phẩm')}**")
                    st.caption(f"Mức độ phù hợp: **{row.score*100:.1f}%**")
                    
                    if is_hybrid:
                        reason = f"Khớp hoàn toàn với gu của bạn về **{aspect_dict_vn[item_strength]}**." if item_strength == user_gu else f"Sản phẩm nổi bật về **{aspect_dict_vn[item_strength]}** đang được nhiều người ưa chuộng."
                    else:
                        reason = f"Được đánh giá cao về **{aspect_dict_vn[item_strength]}** - Khớp với tiêu chí bạn vừa thiết lập."
                    st.write(f"💡 **Lý do:** {reason}")
            valid_count += 1

# ==========================================
# 4. GIAO DIỆN CHÍNH (FRONTEND)
# ==========================================
st.title("📱 Smart Recommendation System")

if not st.session_state['logged_in']:
    # MÀN HÌNH ĐĂNG NHẬP / ĐĂNG KÝ
    tab_login, tab_register = st.tabs(["🔑 Đăng Nhập", "📝 Đăng Ký Tài Khoản Mới"])
    
    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Tên đăng nhập")
            p = st.text_input("Mật khẩu", type="password")
            if st.form_submit_button("Đăng Nhập"):
                if u in st.session_state['user_db'] and st.session_state['user_db'][u]["password"] == p:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = st.session_state['user_db'][u]
                    st.rerun()
                else: st.error("❌ Sai tên đăng nhập hoặc mật khẩu!")
                
    with tab_register:
        st.write("Tạo tài khoản để cá nhân hóa trải nghiệm mua sắm của riêng bạn.")
        with st.form("register_form"):
            new_u = st.text_input("Tên đăng nhập mới *")
            new_p = st.text_input("Mật khẩu *", type="password")
            new_name = st.text_input("Họ và Tên")
            
            st.markdown("---")
            st.markdown("**🛠️ Thiết lập Cấu hình Sở thích (Rất quan trọng)**")
            st.write("Hãy chọn những tiêu chí bạn ưu tiên nhất để AI học hỏi:")
            selections = st.multiselect("Lựa chọn tiêu chí:", options=list(aspect_dict_vn.keys()), format_func=lambda x: aspect_dict_vn[x])
            
            if st.form_submit_button("Xác nhận Đăng Ký"):
                if new_u in st.session_state['user_db']:
                    st.error("❌ Tên đăng nhập này đã tồn tại!")
                elif not new_u or not new_p:
                    st.warning("⚠️ Vui lòng điền đủ Tên đăng nhập và Mật khẩu!")
                elif not selections:
                    st.warning("⚠️ Hãy chọn ít nhất 1 sở thích để AI nhận diện bạn nhé!")
                else:
                    # Chế tạo Custom Vector cho user mới
                    vec = np.zeros((1, 6))
                    for s in selections:
                        vec[0, list(aspect_dict_vn.keys()).index(s)] = 1.0
                    
                    # Lưu user mới vào Database ảo
                    st.session_state['user_db'][new_u] = {
                        "password": new_p,
                        "name": new_name if new_name else new_u,
                        "is_new": True,
                        "custom_vector": vec,
                        "top_aspect": selections[0] # Lấy cái đầu tiên làm gu chính
                    }
                    st.success("🎉 Đăng ký thành công! Vui lòng chuyển sang tab Đăng Nhập.")

else:
    # MÀN HÌNH SAU KHI ĐĂNG NHẬP
    user = st.session_state['current_user']
    
    col_header, col_out = st.columns([5, 1])
    col_header.success(f"Chào mừng trở lại, **{user['name']}**!")
    if col_out.button("Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.rerun()

    st.divider()

    # BỘ ĐỊNH TUYẾN (SMART ROUTER)
    if user['is_new']:
        # Xử lý cho User Đăng ký mới
        top_aspect_vn = aspect_dict_vn[user['top_aspect']]
        st.info(f"✨ **Trải nghiệm Tân binh:** AI đang rà soát kho hàng dựa trên Vector sở thích **{top_aspect_vn}** mà bạn vừa thiết lập.")
        st.markdown("### 🎯 Đề xuất Khởi động nhanh (Content-Based)")
        
        recs = get_content_recommendations(user['custom_vector'])
        render_product_cards(recs, is_hybrid=False)
        
    else:
        # Xử lý cho User cũ đã có trong Data
        real_id = user['real_id']
        user_gu = user_profiles.loc[real_id].idxmax()
        st.info(f"🧠 **Lịch sử AI phân tích:** Hệ thống ghi nhận bạn thường ưu tiên các sản phẩm có chất lượng **{aspect_dict_vn[user_gu]}**.")
        st.markdown("### 🎯 Gợi ý Tối ưu hôm nay (Hybrid AI)")
        
        recs = get_hybrid_recommendations(real_id)
        render_product_cards(recs, is_hybrid=True, user_gu=user_gu)