import json
import nltk
from textblob import TextBlob
from collections import Counter
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


def count_adjectives_in_reviews(adjectives, reviews, positive=True):
    threshold = 0.1 if positive else -0.1
    count = Counter()
    review_count = 0

    for review in reviews:
        review_blob = TextBlob(review)
        if (positive and review_blob.sentiment.polarity >= threshold) or (
                not positive and review_blob.sentiment.polarity <= threshold):
            review_adjectives = filter_adjectives(review_blob.words)
            count.update([adj for adj in review_adjectives if adj in adjectives])
            review_count += 1

    return count, review_count

if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/yelp_academic_dataset_review.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    data = load_data(file_path, target_business_id)
    combined_data = ' '.join(data)
    filtered_data = remove_stopwords(combined_data)
    unique_words = extract_unique_words(filtered_data)
    adjectives = filter_adjectives(unique_words)

    # Count adjectives in positive and negative reviews and total reviews
    positive_adjective_counts, positive_review_count = count_adjectives_in_reviews(adjectives, data, positive=True)
    negative_adjective_counts, negative_review_count = count_adjectives_in_reviews(adjectives, data, positive=False)

    # Get top 5 positive and negative words\
    top_5_positive = positive_adjective_counts.most_common(5)
    top_5_negative = negative_adjective_counts.most_common(5)

    # Print results
    print("Positive Reviews Count:", positive_review_count)
    print("Negative Reviews Count:", negative_review_count)
    print("\nTop 5 Positive Words:")
    for word, count in top_5_positive:
        print(f"{word}: {count}")

    print("\nTop 5 Negative Words:")
    for word, count in top_5_negative:
        print(f"{word}: {count}")
