# chạy bằng lệnh: streamlit run demo2.py  
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from hotel_recommender import HotelRecommenderBERT

#---------------------------------------------------------------------------------------------------------------------#
# Đọc dữ liệu du khách và khách sạn từ file CSV.
new_hotels = pd.read_csv('../hotels_data_final.csv')
#new_tourists = pd.read_csv('../tourist_dataset_10k.csv')

# class khuyến nghị
recommender = HotelRecommenderBERT(new_hotels)
#---------------------------------------------------------------------------------------------------------------------#
# Danh sách các tỉnh/thành phố
locations = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bạc Liêu", "Bắc Giang", "Bắc Kạn", "Bắc Ninh",
    "Bến Tre", "Bình Dương", "Bình Định", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cao Bằng", "Cần Thơ", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình",
    "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La",
    "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang",
    "TP. Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]
#---------------------------------------------------------------------------------------------------------------------#
# Cấu hình trang
st.set_page_config(
    page_title="Advanced Hotel Recommendation System",
    page_icon="🏨",
    layout="wide"
)

# Sidebar - Nhập thông tin
with st.sidebar:
    st.header("🔍 Search Criteria")
    with st.form("search_form"):
        # Thông tin cơ bản
        st.subheader("Basic Information")
        selected_location = st.selectbox("Location", locations, index=locations.index("Hồ Chí Minh") if "Hồ Chí Minh" in locations else 0)
        
        # Budget range
        st.subheader("Budget Range (VND)")
        min_budget, max_budget = st.slider(
            "Select your budget range",
            0, 20000000, (0, 20000000),
            step=100000,
            format="%d VND"
        )
        
        # Time preferences
        st.subheader("Check-in/out Time Preferences")
        checkin_start, checkin_end = st.slider(
            "Preferred check-in time",
            0, 24, (0, 24),
            step=1,
            format="%02d:00"
        )
        
        checkout_start, checkout_end = st.slider(
            "Preferred check-out time",
            0, 24, (0, 24),
            step=1,
            format="%02d:00"
        )
        
        # Input text
        st.subheader("Additional Preferences")
        special_requests = st.text_area("Special requests", placeholder="Tôi cần tìm khách sạn gần biển, điểm đánh giá khoảng 8.0")
        submitted = st.form_submit_button("Find Hotels")
        
        if submitted:
            st.session_state.search_params = {
                'location': selected_location,
                'min_budget': min_budget,
                'max_budget': max_budget,
                'checkin_range': (checkin_start, checkin_end),
                'checkout_range': (checkout_start, checkout_end),
                'special_requests': special_requests
            }
            # Xoá các kết quả hiển thị cũ nếu có
            for key in ['recommendations', 'show_count', 'other_show_count']:
                if key in st.session_state:
                    del st.session_state[key]

            # Gọi rerun để làm mới giao diện
            st.rerun()
            st.success("Search criteria saved!")

# Main content
st.title("🏨 Advanced Hotel Recommendation System")
st.write("Find hotels that match your exact preferences")

if 'search_params' in st.session_state:
    params = st.session_state.search_params
    
    # Hiển thị tiêu chí tìm kiếm
    with st.expander("🔍 Your Search Criteria", expanded=True):
        # Dòng 1: Location + Budget
        cols1 = st.columns(2)
        cols1[0].metric("Location", params['location'])
        cols1[1].metric("Budget", f"{params['min_budget']:,} - {params['max_budget']:,} VND")

        # Chuẩn bị text cho check-in / check-out
        checkin_text = f"{params['checkin_range'][0]}:00 - {params['checkin_range'][1]}:00"
        checkout_text = f"{params['checkout_range'][0]}:00 - {params['checkout_range'][1]}:00"

        # Chuẩn bị text cho requests
        special_requests_text = params['special_requests'] if params['special_requests'] else "Không có"

        # Dòng 2: Facilities + Check-in
        cols2 = st.columns(2)
        cols2[0].markdown(f"**Special Requests:** {special_requests_text}")
        cols2[1].markdown(f"**Check-in Range:** {checkin_text}")

        # Dòng 3: Special Requests + Check-out
        cols3 = st.columns(2)
        cols3[1].markdown(f"**Check-out Range:** {checkout_text}")

    
    # Tìm kiếm khách sạn
    if st.button("🔍 Search Hotels", use_container_width=True):
        with st.spinner(f"Finding best hotels in {params['location']}..."):
            time.sleep(0.5)
            
            # Chuẩn bị dữ liệu đầu vào
            facilities_input = params['special_requests']
                
            # Gọi hàm gợi ý
            recommendations = recommender.get_recommendations_bert(
                input_text=facilities_input,
                location=params['location'],
                checkin_range=params['checkin_range'],
                checkout_range=params['checkout_range'],
                min_budget=params['min_budget'],
                max_budget=params['max_budget'],
            )
            
            st.session_state.recommendations = recommendations
    
    # Hiển thị kết quả
    if 'recommendations' in st.session_state:
        st.subheader("🏆 Matching Hotels")
        if not st.session_state.recommendations.empty:
            st.success(f"Found {len(st.session_state.recommendations)} matching hotels")
            # Initialize session state for pagination if not exists
            if 'show_count' not in st.session_state:
                st.session_state.show_count = 5  # Show first 5 hotels initially
                
            # Display hotels up to the current show_count
            df = st.session_state.recommendations.iloc[:st.session_state.show_count]
            
            for idx, hotel in df.iterrows():
                with st.container():
                    st.markdown(f"### [{hotel['Hotel Name']}]({hotel['Hotel URL']})")
                    st.markdown(f"📍 **{hotel['Address']}**")
                    
                    # Hiển thị rating
                    rating = hotel['Overall Rating']
                    stars = "⭐" * int(round(rating))
                    st.markdown(f"{stars} **{rating:.1f}/10**")
                    
                    # Giá và thời gian
                    cols2 = st.columns(3)
                    cols2[0].metric("Price", hotel['Overview Price'])
                    cols2[1].metric("Check-in", hotel['Checkin Time'])
                    cols2[2].metric("Check-out", hotel['Checkout Time'])
                    
                    # Facilities
                    with st.expander("Facilities & Details"):
                        st.markdown(f"**Popular Facilities:** {hotel['Popular Facilities']}")
                        st.markdown("**Ratings Breakdown:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - Staff: {hotel['Staff']}
                            - Facilities: {hotel['Facilities']}
                            - Cleanliness: {hotel['Cleanliness']}
                            - Comfort: {hotel['Comfort']}
                            """)
                        with col2:
                            st.markdown(f"""
                            - Value for Money: {hotel['Value for Money']}
                            - Location: {hotel['Location']}
                            - Free Wifi: {hotel['Free Wifi']}
                            """)
                    
                    st.markdown("---")
            
            # Show "Load More" button if there are more hotels to display
            if len(st.session_state.recommendations) > st.session_state.show_count:
                if st.button("Load More Hotels"):
                    st.session_state.show_count += 5  # Increase by 5 each time
                    st.rerun()
        else:
            st.warning("No hotels found matching all your criteria. Please try adjusting your search parameters.")   
            
        st.markdown("---")
        st.subheader("🏡 Other Hotels in the Area")
        other_hotels = recommender.get_other_hotels(st.session_state.recommendations, params['location'])
        if not other_hotels.empty:
            st.success(f"Found {len(other_hotels)} other hotels in {params['location']}")
            
            # Initialize session state for other hotels pagination
            if 'other_show_count' not in st.session_state:
                st.session_state.other_show_count = 5  
                
            # Display other hotels up to current show_count
            other_df = other_hotels.iloc[:st.session_state.other_show_count]
            
            for idx, hotel in other_df.iterrows():
                with st.container():
                    st.markdown(f"### [{hotel['Hotel Name']}]({hotel['Hotel URL']})")
                    st.markdown(f"📍 **{hotel['Address']}**")
                    
                    rating = hotel['Overall Rating']
                    stars = "⭐" * int(round(rating))
                    st.markdown(f"{stars} **{rating:.1f}/10**")
                    
                    st.markdown(f"**💰 Price:** {hotel['Overview Price']}")
                    st.markdown(f"**🏨 Check-in Time:** {hotel['Checkin Time']}")
                    st.markdown(f"**🏨 Check-out Time:** {hotel['Checkout Time']}")
                    
                    with st.expander("View Facilities"):
                        st.markdown(f"**Facilities:** {hotel['Popular Facilities']}")
                        st.markdown("**Ratings Breakdown:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - Staff: {hotel['Staff']}
                            - Facilities: {hotel['Facilities']}
                            - Cleanliness: {hotel['Cleanliness']}
                            - Comfort: {hotel['Comfort']}
                            """)
                        with col2:
                            st.markdown(f"""
                            - Value for Money: {hotel['Value for Money']}
                            - Location: {hotel['Location']}
                            - Free Wifi: {hotel['Free Wifi']}
                            """)
                    
                    st.markdown("---")
            
            # Show "Load More" button for other hotels if there are more
            if len(other_hotels) > st.session_state.other_show_count:
                if st.button("Load More Other Hotels"):
                    st.session_state.other_show_count += 5  
                    st.rerun()
        else:
            st.warning(f"No other hotels found in {params['location']}")
        
else:
    st.info("Please set your search criteria in the sidebar to find matching hotels")

# Footer
st.markdown("---")
st.markdown("© 2025 Hotel Recommendation System | Quoc Thai - Khanh Ngan | IE207")
