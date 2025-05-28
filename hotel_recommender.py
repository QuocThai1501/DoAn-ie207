import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode
import re
from datetime import datetime
import time

class HotelRecommenderv1:
    def __init__(self, hotels_df, tourists_df):
        self.hotels_data = hotels_df.copy()
        self.tourists_data = tourists_df.copy()

    # xử lý chuỗi
    def preprocess_text(self, text):
        if pd.isnull(text):
            return ''
        return unidecode(text.lower())  # bỏ dấu, viết thường

    # xử lý giá hotels
    def parse_price(self, price_str):
        try:
            price = price_str.replace(".", "").replace(" VND", "").strip()
            return int(price)
        except:
            return None

    # xử lý khoảng giá tourist
    def parse_budget_range(self, budget_str):
        try:
            parts = budget_str.replace(" VND", "").split("-")
            min_budget = int(parts[0].replace(".", "").strip())
            max_budget = int(parts[1].replace(".", "").strip())
            return min_budget, max_budget
        except:
            return None, None

    # xử lý khoảng giá tourist
    def parse_budget_range(self, budget_str):
        try:
            parts = budget_str.replace(" VND", "").split("-")
            min_budget = int(parts[0].replace(".", "").strip())
            max_budget = int(parts[1].replace(".", "").strip())
            return min_budget, max_budget
        except:
            return None, None

    # chuẩn hóa thời gian
    def parse_time_range(self, text):
        text = text.lower().strip()
        
        if 'phục vụ 24h' in text:
            return (0, 24)

        time_pattern = r'(\d{1,2}:\d{2})'

        times = re.findall(time_pattern, text)
        times = [datetime.strptime(t, "%H:%M").hour for t in times]

        if 'từ' in text and 'đến' not in text and len(times) == 1:
            return (times[0], 24)
        elif 'đến' in text and 'từ' not in text and len(times) == 1:
            return (0, times[0])
        elif len(times) == 2:
            if (times[0] == times[1]): return (0, 24)
            return (times[0], times[1])
        else:
            return (0, 24)

    # xử lý khoảng thời gian    
    def time_ranges_overlap(range1, range2):
        start1, end1 = range1
        start2, end2 = range2
        return max(start1, start2) < min(end1, end2)



    def get_recommendations_tfidf(self, touristid, location):
        hotel_data = self.hotels_data.copy()
        tourist_data = self.tourists_data.copy()

    # Kết hợp 3 cột thông tin của tourist thành 1 chuỗi văn bản
        def combine_text(row):
            return f"{row['requirements']} {row['preferred_facilities']} {row['special_requests']}"

    # Tiền xử lý dữ liệu
        hotel_data['Popular Facilities Original'] = hotel_data['Popular Facilities'] # giữ lại giá trị có dấu
        hotel_data['Popular Facilities'] = hotel_data['Popular Facilities'].apply(lambda x: self.preprocess_text(x))

        tourist_data['combined_text'] = tourist_data.apply(combine_text, axis=1)
        tourist_data['combined_text'] = tourist_data['combined_text'].apply(lambda x: self.preprocess_text(x))


    # Tạo TF-IDF matrix cho cả tourist và hotel
        corpus = tourist_data['combined_text'].tolist() + hotel_data['Popular Facilities'].tolist()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

    # Chia ma trận lại thành 2 phần
        tfidf_tourist = tfidf_matrix[:len(tourist_data)]
        tfidf_hotel = tfidf_matrix[len(tourist_data):]

    # Tính độ tương đồng
        cosine_sim = linear_kernel(tfidf_tourist, tfidf_hotel)

    # Tìm index tourist theo ID
        tourist_idx = tourist_data[tourist_data['id'] == touristid].index
        if len(tourist_idx) == 0:
            print(f"Không tìm thấy user với ID {touristid}")
            return pd.DataFrame()
        i = tourist_idx[0]
    # Lấy checkin & checkout time của tourist
        # tourist_checkin = tourist_data.iloc[i].get('checkin_time_preference')
        # tourist_checkout = tourist_data.iloc[i].get('checkout_time_preference')
        # tourist_checkin_range = parse_time_range(tourist_checkin)
        # tourist_checkout_range = parse_time_range(tourist_checkout)
        
    # Lấy ngân sách của tourist
        budget_str = tourist_data.iloc[i]['budget_range']
        min_budget, max_budget = self.parse_budget_range(budget_str)

        # Lấy top hotel phù hợp nhất
        sim_scores = list(enumerate(cosine_sim[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:500]  # loại bỏ bản thân

        job_indices = [score[0] for score in sim_scores]

    # Lọc theo location và tạo danh sách đề xuất
        recommendations = []
        for idx in job_indices:
            row = hotel_data.iloc[idx]

            # kiểm tra location (tỉnh)
            if hotel_data.iloc[idx]['Province'].lower() == location.lower():
                
                # hotel_checkin_range = parse_time_range(row['Checkin Time'])
                # hotel_checkout_range = parse_time_range(row['Checkout Time'])

                # if (time_ranges_overlap(tourist_checkin_range, hotel_checkin_range) and
                #      time_ranges_overlap(tourist_checkout_range, hotel_checkout_range)):
                price = self.parse_price(row['Overview Price'])

                # kiểm tra khoảng giá
                if  price <= max_budget:
                    recommendations.append([
                        row['Hotel URL'], row['Hotel Name'], row['Overview Price'], row['Address'],
                        row['Overall Rating'], row['Staff'], row['Facilities'], row['Cleanliness'],
                        row['Comfort'], row['Value for Money'], row['Location'], row['Free Wifi'],
                        row['Popular Facilities Original'], row['Checkin Time'], row['Checkout Time'], row['Province']
                    ])
            # if len(recommendations) == num_hotels:
            #     break

    # Tạo DataFrame kết quả
        columns = ['Hotel URL', 'Hotel Name', 'Overview Price', 'Address', 'Overall Rating', 'Staff', 'Facilities', 'Cleanliness', 
                'Comfort', 'Value for Money', 'Location', 'Free Wifi', 'Popular Facilities', 'Checkin Time', 'Checkout Time', 'Province']
        return pd.DataFrame(recommendations, columns=columns)



    # Lấy các khách sạn khác trong cùng tỉnh nhưng không nằm trong recommendations
    def get_other_hotels(self, recommendations, location):
        recommended_urls = recommendations['Hotel URL']  # Hotel URL đã đề xuất
        hotel_data = self.hotels_data.copy()
        print('số hotel: ', recommended_urls)
        other_hotels = hotel_data[
            (hotel_data['Province'].str.lower() == location.lower()) &
            (~hotel_data['Hotel URL'].isin(recommended_urls))
        ]

        # Tạo danh sách các khách sạn khác
        other_rows = other_hotels[[
            'Hotel URL', 'Hotel Name', 'Overview Price', 'Address',
            'Overall Rating', 'Staff', 'Facilities', 'Cleanliness',
            'Comfort', 'Value for Money', 'Location', 'Free Wifi',
            'Popular Facilities', 'Checkin Time', 'Checkout Time', 'Province'
        ]].values.tolist()

    # Tạo DataFrame kết quả
        columns = ['Hotel URL', 'Hotel Name', 'Overview Price', 'Address', 'Overall Rating', 'Staff', 'Facilities', 'Cleanliness', 
                'Comfort', 'Value for Money', 'Location', 'Free Wifi', 'Popular Facilities', 'Checkin Time', 'Checkout Time', 'Province']
        return pd.DataFrame(other_rows, columns=columns)