import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode
import re
from datetime import datetime
import os, torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class BaseHotel: # class cha cho 2 version demo 1 và 2
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
    def time_ranges_overlap(self, range1, range2):
        start1, end1 = range1
        start2, end2 = range2
        return max(start1, start2) < min(end1, end2)

 # Lấy các khách sạn khác trong cùng tỉnh nhưng không nằm trong recommendations
    def get_other_hotels(self, recommendations, location):
        hotel_data = self.hotels_data.copy()

        if recommendations is None or recommendations.empty:
            # Nếu không có recommendations, trả về tất cả hotels trong tỉnh
            other_hotels = hotel_data[
                hotel_data['Province'].str.lower() == location.lower()
            ]
        else:
            recommended_urls = recommendations['Hotel URL']
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



class HotelRecommenderv1(BaseHotel):
    def __init__(self, hotels_df, tourists_df):
        self.hotels_data = hotels_df.copy()
        self.tourists_data = tourists_df.copy()



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

    # Tạo DataFrame kết quả
        columns = ['Hotel URL', 'Hotel Name', 'Overview Price', 'Address', 'Overall Rating', 'Staff', 'Facilities', 'Cleanliness', 
                'Comfort', 'Value for Money', 'Location', 'Free Wifi', 'Popular Facilities', 'Checkin Time', 'Checkout Time', 'Province']
        return pd.DataFrame(recommendations, columns=columns)

    

class HotelRecommenderv2(BaseHotel):
    def __init__(self, hotels_df, tourists_df):
        self.hotels_data = hotels_df.copy()
        self.tourists_data = tourists_df.copy()

    

    def get_recommendations_tfidf_input(self, facilities_input, location, checkin_range, checkout_range, min_budget, max_budget):
        hotel_data = self.hotels_data.copy()
    
        # Tiền xử lý dữ liệu
        combined_text = ' '.join([str(f) for f in facilities_input if pd.notna(f)])
        hotel_data['Popular Facilities Original'] = hotel_data['Popular Facilities']
        hotel_data['Popular Facilities'] = hotel_data['Popular Facilities'].apply(lambda x: self.preprocess_text(x))
        combined_text = self.preprocess_text(combined_text)

        # Tạo TF-IDF matrix
        corpus = [combined_text] + hotel_data['Popular Facilities'].tolist()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Tính toán độ tương đồng
        cosine_sim = linear_kernel(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for idx, score in sim_scores:
            row = hotel_data.iloc[idx]
            
            # Kiểm tra location
            if row['Province'].lower() != location.lower():
                continue
                
            # Kiểm tra giá
            price = self.parse_price(row['Overview Price'])
            if not (min_budget <= price <= max_budget):
                continue
                
            # Kiểm tra thời gian check-in/out
            hotel_checkin = self.parse_time_range(row['Checkin Time'])
            hotel_checkout = self.parse_time_range(row['Checkout Time'])
            
            if not self.time_ranges_overlap(hotel_checkin, checkin_range):
                continue
            if not self.time_ranges_overlap(hotel_checkout, checkout_range):
                continue
                
            recommendations.append({
                'Hotel URL': row['Hotel URL'],
                'Hotel Name': row['Hotel Name'],
                'Overview Price': row['Overview Price'],
                'Address': row['Address'],
                'Overall Rating': row['Overall Rating'],
                'Staff': row['Staff'],
                'Facilities': row['Facilities'],
                'Cleanliness': row['Cleanliness'],
                'Comfort': row['Comfort'],
                'Value for Money': row['Value for Money'],
                'Location': row['Location'],
                'Free Wifi': row['Free Wifi'],
                'Popular Facilities': row['Popular Facilities Original'],
                'Checkin Time': row['Checkin Time'],
                'Checkout Time': row['Checkout Time'],
                'Province': row['Province'],
                'Similarity Score': score,
                'Parsed Price': price
            })

        return pd.DataFrame(recommendations)



class HotelRecommenderBERT(BaseHotel):
    def __init__(self, hotels_df, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.hotels_data = hotels_df.copy()
        self.model = SentenceTransformer(model_name)
        self.hotel_embeddings = self.load_embeddings('hotel_embeddings.npy')

    def load_embeddings(self, file_path='hotel_embeddings.npy'):
        if os.path.exists(file_path):
            embeddings = torch.tensor(np.load(file_path))
        else:
            embeddings = self.encode_hotels()
            np.save(file_path, embeddings.cpu().numpy())
        return embeddings

    def encode_hotels(self):
        def generate_hotel_text(row):
            # Chỉ tập trung encode tiện ích (địa chỉ + rating đã lọc trước rồi)
            text = f"có {row['Popular Facilities']}"
            return text

        # Áp dụng lên toàn bộ DataFrame
        texts = self.hotels_data.apply(generate_hotel_text, axis=1)
        
        # Encode thành embedding tensor
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def extract_address_phrases(self, text):
        patterns = [
            r"(quận\s+[a-zàáảãạăâđêếểễệôơưùúủũụưăâêôơế\d\-]+)",
            r"(phường\s+[a-zàáảãạăâđêếểễệôơưùúủũụưăâêôơế\d\-]+)",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text.lower()))
        return matches

    def split_address_rating_features(self, input_text, address_keywords, rating_keywords):
        tokens = [t.strip() for t in input_text.lower().split(',')]

        address_parts = []
        rating_parts = []
        feature_parts = []

        for token in tokens:
            if any(kw in token for kw in address_keywords):
                address_parts.append(token)
            elif any(kw in token for kw in rating_keywords):
                rating_parts.append(token)
            else:
                feature_parts.append(token)

        return ' '.join(address_parts), ' '.join(rating_parts), ' '.join(feature_parts)

    def get_recommendations_bert(self, input_text, location, checkin_range, checkout_range, min_budget, max_budget):
        hotel_data = self.hotels_data.copy()

        address_keywords = ['số', 'quận', 'phường', 'đường', 'tp', 'thành phố', 'ward', 'district', 'city']
        rating_keywords = ['xuất sắc', 'tốt', 'trung bình', 'kém']

        # Tách input thành 3 cụm: địa chỉ, đánh giá, tiện ích
        address_part, rating_part, feature_part = self.split_address_rating_features(input_text, address_keywords, rating_keywords)

        # ======= LỌC ĐỊA CHỈ =======
        address_phrases = self.extract_address_phrases(address_part)
        if address_phrases:
            pattern = '|'.join(map(re.escape, address_phrases))
            hotel_data = hotel_data[hotel_data['Address'].str.lower().str.contains(pattern)]

        # ======= LỌC ĐÁNH GIÁ =======
        if 'xuất sắc' in rating_part:
            hotel_data = hotel_data[hotel_data['Overall Rating'] >= 9]
        elif 'tốt' in rating_part:
            hotel_data = hotel_data[(hotel_data['Overall Rating'] >= 8) & (hotel_data['Overall Rating'] < 9)]
        elif 'trung bình' in rating_part:
            hotel_data = hotel_data[(hotel_data['Overall Rating'] >= 6) & (hotel_data['Overall Rating'] < 8)]
        elif 'kém' in rating_part:
            hotel_data = hotel_data[hotel_data['Overall Rating'] < 6]

        # ======= TÍNH SIMILARITY CHO TIỆN ÍCH =======
        if feature_part.strip():
            input_text_embedding = self.model.encode(feature_part, convert_to_tensor=True)
            filtered_embeddings = self.hotel_embeddings[hotel_data.index]
            cosine_scores = util.pytorch_cos_sim(input_text_embedding, filtered_embeddings)[0]
            cosine_scores_np = cosine_scores.cpu().numpy()
        else:
            cosine_scores_np = np.ones(len(hotel_data))  # Nếu không có tiện ích, similarity mặc định = 1

        matched_hotels = []

        # ======= VÒNG LẶP LỌC CHI TIẾT =======
        for idx, (_, row) in enumerate(hotel_data.iterrows()):
            similarity = cosine_scores_np[idx]
            if similarity <= 0.5:
                continue  # bỏ những khách sạn tương đồng thấp

            # Lọc theo location (tỉnh/thành phố)
            if row['Province'].lower() != location.lower():
                continue

            # Lọc theo giá
            price = self.parse_price(row['Overview Price'])
            if price is None or not (min_budget <= price <= max_budget):
                continue

            # Lọc theo check-in time
            hotel_checkin = self.parse_time_range(row['Checkin Time'])
            if not self.time_ranges_overlap(hotel_checkin, checkin_range):
                continue

            # Lọc theo check-out time
            hotel_checkout = self.parse_time_range(row['Checkout Time'])
            if not self.time_ranges_overlap(hotel_checkout, checkout_range):
                continue

            # Giữ lại kết quả
            row_copy = row.copy()
            row_copy['similarity_score'] = similarity
            matched_hotels.append(row_copy)

        # Chuyển kết quả thành DataFrame
        if matched_hotels:
            result_df = pd.DataFrame(matched_hotels)
            result_df = result_df.sort_values(by='similarity_score', ascending=False)
        else:
            result_df = pd.DataFrame(columns=list(self.hotels_data.columns) + ['similarity_score'])

        return result_df

