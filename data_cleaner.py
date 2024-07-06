import pandas as pd
# import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# # Download the NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

comments_data = pd.read_csv("data/comments.csv")
posts_data = pd.read_csv("data/posts.csv")

bad_posts = [
    "1dtmogb",
    "1aof69c",
    "1bapah6",
    "18i0lqf",
    "116a9qq",
    "146pc14",
    "16pp2h6",
    "1cja071",
    "106yqxk",
    "1cqx1iz",
    "1atdd05",
    "1527ivl",
    "17k8rsn",
    "14dew25",
    "15b34ng",
    "16ofae7",
    "13l893a",
    "13dufss",
    "175s6nt",
    "16xh795",
    "1cksd4x",
    "133gqee",
    "11yiygr",
    "13r3v36",
    "150sshi",
    "159yd7n",
    "1cfw1ra",
    "172dubs",
    "12s31nh",
    "12qhb8x",
    "18501uz",
    "1czg3x6",
    "15rtqgc",
    "164jmfw",
    "13l81jl",
    "1675zhp",
    "13r3v36",
    "1808tpw",
    "18qbdzh",
    "1beimov",
    "112fp9z",
]

# Remove bad posts
comments_data = comments_data[~comments_data["post_id"].isin(bad_posts)]
posts_data = posts_data[~posts_data["post_id"].isin(bad_posts)]

# Save the cleaned data
comments_data.to_csv("data/comments_cleaned.csv", index=False)
posts_data.to_csv("data/posts_cleaned.csv", index=False)


def clean_text(text):
    """
    Function to clean the text data
    Args: text (str): Text data to be cleaned
    Returns: list: List of cleaned words
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    # Add chatgpt, university, college, professor, exam to stopwords (for exploratory analysis)
    # stop_words.update(['chatgpt', 'university', 'college', 'professor', 'exam', 'ai'])
    text = [word for word in text if word not in stop_words]
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    text = [
        (
            lemmatizer.lemmatize(i, j[0].lower())
            if j[0].lower() in ["a", "n", "v"]
            else lemmatizer.lemmatize(i)
        )
        for i, j in pos_tag(text)
    ]
    return text
