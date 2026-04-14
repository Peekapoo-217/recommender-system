# ==========================================
# GIAO DIỆN CHÍNH (Chia 2 Tab)
# ==========================================
st.title("📱 Smart Recommendation System")

# Tạo 2 Tab cho 2 đối tượng khách hàng
tab1, tab2 = st.tabs(["👤 Khách hàng thân thiết (Đăng nhập)", "✨ Khách hàng mới (Trải nghiệm AI)"])

# ----------------------------------------------------
# TAB 1: DÀNH CHO KHÁCH HÀNG CŨ (Hệ thống lai Hybrid)
# ----------------------------------------------------
with tab1:
    if not st.session_state['logged_in']:
        st.write("Vui lòng đăng nhập để AI phân tích lịch sử mua sắm của bạn.")
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập (Thử: admin, khachhang1)")
            password = st.text_input("Mật khẩu (Thử: 123)", type="password")
            submitted = st.form_submit_button("Đăng Nhập")
            if submitted:
                if username in USER_DB and USER_DB[username]["password"] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = USER_DB[username]
                    st.rerun()
                else:
                    st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng!")
    else:
        current_user = st.session_state['current_user']
        real_id = current_user['real_id']
        
        col_greet, col_logout = st.columns([3, 1])
        with col_greet:
            st.success(f"👋 Chào mừng trở lại, **{current_user['name']}**!")
        with col_logout:
            if st.button("Đăng xuất", key="logout_btn"):
                st.session_state['logged_in'] = False
                st.rerun()
                
        user_aspect_profile = user_profiles.loc[real_id]
        top_user_aspect = user_aspect_profile.idxmax()
        top_user_aspect_vn = aspect_dict_vn.get(top_user_aspect, top_user_aspect)
        
        st.info(f"🧠 **Lịch sử AI phân tích:** Bạn có xu hướng khắt khe và quan tâm nhiều nhất đến **{top_user_aspect_vn}**.")
        st.markdown("### 🎯 Gợi ý hôm nay cho bạn")
        
        recs = get_hybrid_recommendations(real_id, alpha=0.2, top_n=15) # Lấy dư ra 15 cái để lọc
        
        # LOGIC LỌC SẢN PHẨM VÔ DANH
        valid_recs = []
        for row in recs.itertuples():
            if row.ASIN in product_info: # CHỈ lấy sản phẩm có trong Metadata
                valid_recs.append(row)
            if len(valid_recs) == 5:     # Đủ 5 cái thì dừng
                break
                
        for rank, row in enumerate(valid_recs, 1):
            asin = row.ASIN
            score = row.hybrid_score
            item_meta = product_info[asin]
            title = item_meta.get('title', "Sản phẩm")
            img_url = item_meta.get('imUrl')
            
            top_item_aspect = item_profiles.loc[asin].idxmax()
            top_item_aspect_vn = aspect_dict_vn.get(top_item_aspect, top_item_aspect)
            
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                with c1:
                    if img_url: st.image(img_url)
                    else: st.write("📦")
                with c2:
                    st.markdown(f"**Top {rank}: {title}**")
                    st.caption(f"Độ tự tin: **{score * 100:.1f}%**")
                    if top_user_aspect == top_item_aspect:
                        st.write(f"✅ **Lý do đề xuất:** Cực kỳ xuất sắc về **{top_item_aspect_vn}**, khớp 100% với gu mua sắm của bạn.")
                    else:
                        st.write(f"🌟 **Lý do đề xuất:** Nổi bật về **{top_item_aspect_vn}**, đang lọt top thịnh hành cùng nhóm khách hàng giống bạn.")

# ----------------------------------------------------
# TAB 2: DÀNH CHO KHÁCH HÀNG MỚI (Giải quyết Cold-Start)
# ----------------------------------------------------
with tab2:
    st.markdown("### 🛠️ Thiết lập Sở thích Cá nhân (Cold-Start Problem)")
    st.write("Vì bạn là người dùng mới, hãy cho AI biết bạn quan tâm đến điều gì nhất khi mua phụ kiện điện thoại nhé!")
    
    # Cho người dùng chọn các tiêu chí
    selected_aspects = st.multiselect(
        "Chọn 1 hoặc nhiều tiêu chí quan trọng nhất với bạn:",
        options=list(aspect_dict_vn.keys()),
        format_func=lambda x: aspect_dict_vn[x]
    )
    
    if st.button("🔍 Tìm sản phẩm phù hợp", type="primary"):
        if not selected_aspects:
            st.warning("Vui lòng chọn ít nhất 1 tiêu chí!")
        else:
            # 1. Chế tạo Vector Sở thích tức thời
            # Mặc định tất cả bằng 0, cái nào user chọn thì cho điểm tuyệt đối 1.0
            custom_vector = np.zeros((1, 6))
            for aspect in selected_aspects:
                idx = list(aspect_dict_vn.keys()).index(aspect)
                custom_vector[0, idx] = 1.0 
                
            # 2. So khớp Vector này với toàn bộ kho sản phẩm (Content-Based thuần túy)
            sim_scores = cosine_similarity(custom_vector, item_matrix).flatten()
            sim_scores_norm = (sim_scores + 1) / 2 # Chuẩn hóa về [0, 1]
            
            # 3. Lấy Top sản phẩm có thông tin
            new_user_df = pd.DataFrame({'ASIN': all_items, 'score': sim_scores_norm})
            new_user_recs = new_user_df.sort_values(by='score', ascending=False)
            
            valid_new_recs = []
            for row in new_user_recs.itertuples():
                if row.ASIN in product_info:
                    valid_new_recs.append(row)
                if len(valid_new_recs) == 5:
                    break
                    
            # 4. Hiển thị kết quả
            st.divider()
            st.success("🎉 Đã tìm thấy các sản phẩm phù hợp nhất với cấu hình của bạn!")
            
            for rank, row in enumerate(valid_new_recs, 1):
                asin = row.ASIN
                item_meta = product_info[asin]
                title = item_meta.get('title', "Sản phẩm")
                img_url = item_meta.get('imUrl')
                
                # Tìm xem sản phẩm này mạnh nhất môn nào trong các môn mà user vừa chọn
                item_scores = item_profiles.loc[asin]
                best_match_aspect = None
                best_match_score = -99
                for asp in selected_aspects:
                    if item_scores[asp] > best_match_score:
                        best_match_score = item_scores[asp]
                        best_match_aspect = asp
                
                match_aspect_vn = aspect_dict_vn[best_match_aspect]
                
                with st.container(border=True):
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        if img_url: st.image(img_url)
                        else: st.write("📦")
                    with c2:
                        st.markdown(f"**Top {rank}: {title}**")
                        st.caption(f"Mức độ tương thích cấu hình: **{row.score * 100:.1f}%**")
                        st.write(f"🎯 **Tại sao AI chọn?:** Sản phẩm này được cộng đồng mạng đánh giá cực cao về **{match_aspect_vn}** - chính là tiêu chí bạn đang ưu tiên!")