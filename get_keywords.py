import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')



# Load Yelp review dataset
def load_data(file_path, target_business_id):
    with open(file_path, "r") as f:
        lines = f.readlines()

    reviews = []
    for line in lines:
        review = json.loads(line)
        if review["business_id"] == target_business_id:
            reviews.append(review["text"])

    return reviews


# Compute TF-IDF
def compute_tfidf(corpus):
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(corpus), vectorizer


# Extract keywords based on the highest TF-IDF scores
def extract_keywords(tfidf_matrix, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    row_data = tfidf_matrix.tocoo()
    top_features_indices = row_data.col[np.argsort(-row_data.data)[:top_n]]
    keywords = [feature_names[i] for i in top_features_indices]
    return keywords


if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/yelp_academic_dataset_review.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    data = load_data(file_path, target_business_id)
    combined_data = ' '.join(data)
    tfidf_matrix, vectorizer = compute_tfidf([combined_data])
    keywords = extract_keywords(tfidf_matrix, vectorizer)

    # Print top 10 keywords
    print(f"Top 10 keywords: {', '.join(keywords)}")
