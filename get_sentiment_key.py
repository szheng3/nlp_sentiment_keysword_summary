import json
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

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

# Count the number of reviews containing the specified words
def count_reviews_containing_words(reviews, words):
    counts = {word: 0 for word in words}
    for text in reviews:
        for word in words:
            if word in text:
                counts[word] += 1
    return counts

if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/yelp_academic_dataset_review.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    data = load_data(file_path, target_business_id)
    combined_data = ' '.join(data)
    filtered_data = remove_stopwords(combined_data)
    unique_words = extract_unique_words(filtered_data)
    adjectives = filter_adjectives(unique_words)

    review_counts = count_reviews_containing_words(data, adjectives)
    sorted_adjectives = sorted(review_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 Adjectives by Review Count:")
    for word, count in sorted_adjectives:
        print(f"{word}: {count} reviews")
