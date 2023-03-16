import json
import nltk
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
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

# Remove stopwords from the text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Extract unique words from the text
def extract_unique_words(text):
    words = set(text.split())
    return list(words)

# Filter adjectives from the list of unique words
def filter_adjectives(words):
    adjectives = []
    for word in words:
        blob = TextBlob(word)
        if blob.tags and blob.tags[0][1].startswith('JJ'):
            adjectives.append(word)
    return adjectives

# Extract keywords using BERT and K-means clustering
def extract_keywords_bert(words, model_name, n_clusters=10):
    model = SentenceTransformer(model_name)
    word_embeddings = model.encode(words)

    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(word_embeddings)

    cluster_centers = clustering_model.cluster_centers_

    # Find the closest word in each cluster
    keywords = []
    for cluster_center in cluster_centers:
        distances = np.linalg.norm(word_embeddings - cluster_center, axis=1)
        closest_word_index = np.argmin(distances)
        keywords.append(words[closest_word_index])

    return keywords

if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/yelp_academic_dataset_review.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    data = load_data(file_path, target_business_id)
    combined_data = ' '.join(data)
    filtered_data = remove_stopwords(combined_data)
    unique_words = extract_unique_words(filtered_data)
    adjectives = filter_adjectives(unique_words)

    # Use a pre-trained BERT model (e.g., 'distilbert-base-nli-mean-tokens')
    model_name = 'distilbert-base-nli-mean-tokens'
    keywords = extract_keywords_bert(adjectives, model_name)

    # Print keywords
    print(f"Keywords: {', '.join(keywords)}")
