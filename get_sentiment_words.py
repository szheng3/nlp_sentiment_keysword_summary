import json
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

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

# Extract positive and negative words from the text
def extract_sentiment_words(text):
    words = nltk.word_tokenize(text)
    positive_words = []
    negative_words = []

    for word in words:
        # Skip stop words
        if word.lower() in stopwords.words('english'):
            continue

        blob = TextBlob(word)
        sentiment = blob.sentiment.polarity

        if sentiment > 0.5:
            print(word, sentiment)
            positive_words.append(word)
        elif sentiment < -0.5:
            print(word, sentiment)
            negative_words.append(word)

    return positive_words, negative_words

if __name__ == "__main__":
    file_path = "/Users/shuai/PycharmProjects/NLP_project/yelp_dataset/small.json"  # Replace with your Yelp dataset path
    target_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"  # The business_id you want to extract reviews for
    data = load_data(file_path, target_business_id)
    combined_data = ' '.join(data)
    positive_words, negative_words = extract_sentiment_words(combined_data)

    # Print positive and negative words
    print(f"Positive words: {', '.join(positive_words)}")
    print(f"Negative words: {', '.join(negative_words)}")