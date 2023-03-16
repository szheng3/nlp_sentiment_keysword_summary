import json
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load Yelp review dataset and calculate sentiment scores
def load_data(file_path, target_business_id):
    with open(file_path, "r") as f:
        lines = f.readlines()

    reviews = []
    for line in lines:
        review = json.loads(line)
        if review["business_id"] == target_business_id:
            reviews.append((review["text"], TextBlob(review["text"]).sentiment.polarity))

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

# Get the sentiment polarity score for each word
def get_sentiment_scores(words):
    sentiment_scores = {}
    for word in words:
        blob = TextBlob(word)
        sentiment_scores[word] = blob.sentiment.polarity
    return sentiment_scores

# Count the number of reviews containing the specified words
def count_reviews_containing_words(reviews, words):
    counts = {word: 0 for word in words}
    for text, sentiment in reviews:
        for word in words:
            if word in text:
                counts[word] += 1
    return counts

if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/yelp_academic_dataset_review.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    reviews = load_data(file_path, target_business_id)
    combined_data = ' '.join(text for text, _ in reviews)
    filtered_data = remove_stopwords(combined_data)
    unique_words = extract_unique_words(filtered_data)

    sentiment_scores = get_sentiment_scores(unique_words)

    positive_words = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    negative_words = sorted(sentiment_scores.items(), key=lambda x: x[1])[:5]

    positive_word_counts = count_reviews_containing_words(reviews, [word for word, _ in positive_words])
    negative_word_counts = count_reviews_containing_words(reviews, [word for word, _ in negative_words])

    print("Top 5 Positive Words:")
    for word, sentiment in positive_words:
        print(f"{word}: {sentiment} (found in {positive_word_counts[word]} reviews)")

    print("\nTop 5 Negative Words:")
    for word, sentiment in negative_words:
        print(f"{word}: {sentiment} (found in {negative_word_counts[word]} reviews)")
