# chạy bằng lệnh: streamlit run demo1.py  
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from hotel_recommender import HotelRecommenderv1

#---------------------------------------------------------------------------------------------------------------------#
# Đọc dữ liệu du khách và khách sạn từ file CSV.
new_hotels = pd.read_csv('../hotels_data_final.csv')
new_tourists = pd.read_csv('../tourist_dataset_10k.csv')

# class khuyến nghị
recommender = HotelRecommenderv1(new_hotels, new_tourists)
#---------------------------------------------------------------------------------------------------------------------#
# Cấu hình trang
st.set_page_config(
    page_title="Hotel Recommendation System",
    page_icon="🏨",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("👤 User Information")
    user_id_input = st.text_input("Enter User ID", "")

    # Ràng buộc: Chỉ khi nhập số nguyên mới xử lý
    if user_id_input.strip().isdigit():
        user_id = int(user_id_input)
        user_info = new_tourists[new_tourists['id'] == user_id]
        if not user_info.empty:
            user_info = user_info.iloc[0]
            st.markdown("---")
            with st.expander("👤 User Profile"):
                st.markdown(f"""
                - **Name:** {user_info['name']}
                - **Budget:** {user_info['budget_range']}
                - **Preferred Location:** {user_info['preferred_location']}
                - **Requirements:** {user_info['requirements']}
                - **Check-in Preference:** {user_info['checkin_time_preference']}
                - **Check-out Preference:** {user_info['checkout_time_preference']}
                - **Special Requests:** {user_info['special_requests'] if pd.notna(user_info['special_requests']) else 'None'}
                """)
        else:
            st.error("User ID not found.")
            user_id = None
    else:
        st.info("Please enter a valid numeric User ID.")
        user_id = None

# Main content
st.title("🏨 Hotel Recommendation System")
st.header("Find your perfect stay")


if user_id is not None:
    location = user_info['preferred_location'] # lấy địa điểm yêu thích của du khách

    with st.expander("🔍 Your Search Criteria", expanded=True):
        cols = st.columns(2)
        cols[0].metric("Location", user_info['preferred_location'])
        cols[1].metric("Budget", f"{user_info['budget_range']}") # lấy khoảng ngân sách phù hợp với du khách
        # cols[2].metric("Check-in", f"{user_info['checkin_time_preference']}")
        # cols[3].metric("Check-out", f"{user_info['checkout_time_preference']}")
        
        requirements_text = user_info['requirements'] if pd.notna(user_info['requirements']) else "Không có"
        st.write(f"**Requirements:** {requirements_text}")
        # Hiển thị facilities - nếu không có thì ghi "Không có"
        facilities_text = user_info['preferred_facilities'] if pd.notna(user_info['preferred_facilities']) else "Không có"
        st.write(f"**Facilities:** {facilities_text}")
        
        # Hiển thị special requests - nếu không có thì ghi "Không có"
        special_requests_text = user_info['special_requests'] if pd.notna(user_info['special_requests']) else 'Không có'
        st.write(f"**Special Requests:** {special_requests_text}")

    if st.button("🔍 Search Hotels", use_container_width=True):
        with st.spinner(f"Finding the best hotels in {location}..."):
            time.sleep(0.5)
            df_recommendation = recommender.get_recommendations_tfidf(user_id, location)
            st.session_state.recommendations = df_recommendation
           
        
    if 'recommendations' in st.session_state:
        st.subheader("🏆 Matching Hotels")
        if not st.session_state.recommendations.empty:
        # Hiển thị thông tin user thực tế
            st.success(f"Found {len(st.session_state.recommendations)} matching hotels")
            # Initialize session state for pagination if not exists
            if 'show_count' not in st.session_state:
                st.session_state.show_count = 5  # Show first 5 hotels initially
                
            # Display hotels up to the current show_count
            df = st.session_state.recommendations.iloc[:st.session_state.show_count]
            # Hiển thị khách sạn
            for i, hotel in df.iterrows():
                with st.container():
                    
                    st.markdown(f"### [{hotel['Hotel Name']}]({hotel['Hotel URL']})")
                    st.markdown(f"📍 **{hotel['Address']}**")
                        
                    # Hiển thị rating
                    rating = hotel['Overall Rating'] # điểm đánh giá chung
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
            st.warning("No hotels found matching your criteria")

        st.markdown("---")
        st.subheader("🏡 Other Hotels in the Area")
        other_hotels = recommender.get_other_hotels(st.session_state.recommendations, location)
        if not other_hotels.empty:
            
            # Hiển thị khách sạn khác
            st.success(f"Found {len(other_hotels)} other hotels in {location}")
            # Initialize session state for other hotels pagination
            if 'other_show_count' not in st.session_state:
                st.session_state.other_show_count = 5  
                
            # Display other hotels up to current show_count
            other_df = other_hotels.iloc[:st.session_state.other_show_count]
            for i, hotel in other_df.iterrows():
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
            # Show "Load More" button for other hotels if there are more
            if len(other_hotels) > st.session_state.other_show_count:
                if st.button("Load More Other Hotels"):
                    st.session_state.other_show_count += 5  
                    st.rerun()
        else:
            st.warning("No hotels found matching your criteria")

# Footer
st.markdown("---")
st.markdown("© 2025 Hotel Recommendation System | Quoc Thai - Khanh Ngan | IE207")