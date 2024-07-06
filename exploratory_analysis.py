from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from data_cleaner import clean_text

posts_data = pd.read_csv("data/posts_cleaned.csv")
comments_data = pd.read_csv("data/comments_cleaned.csv")
comments_data = comments_data.merge(posts_data[["post_id", "subreddit"]], on="post_id")

# Number of posts and comments
print(
    "The dataset consists of",
    posts_data.shape[0],
    "posts and",
    comments_data.shape[0],
    "comments",
)

# Check the number of posts and comments per subreddit
print(posts_data["subreddit"].value_counts())
print(comments_data["subreddit"].value_counts())

# Calculate the average number of words in comments
comments_data["num_words"] = comments_data["comment"].apply(lambda x: len(x.split()))
print("The average number of words in comments is", comments_data["num_words"].mean())

# Calculate the average number of words in comments per subreddit
print(comments_data.groupby("subreddit")["num_words"].mean())

# Clean posts and comments
posts_data["cleaned_post"] = posts_data["post_title"].apply(clean_text)
comments_data["cleaned_comment"] = comments_data["comment"].apply(clean_text)

# Calculate the frequency distribution of words in posts' titles
all_words = [word for post in posts_data["cleaned_post"] for word in post]
fdist_posts = FreqDist(all_words)

# Create a word cloud for posts' titles
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(fdist_posts)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("results/wordcloud_all_posts.png")

# Calculate the frequency distribution of words in comments
all_words = [word for comment in comments_data["cleaned_comment"] for word in comment]
fdist_comm = FreqDist(all_words)

# Create word cloud for comments
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(fdist_comm)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("results/wordcloud_all_comments.png")

# Calculate tf-idf of words in comments per subreddit
comments_data["comment_text"] = comments_data["cleaned_comment"].apply(
    lambda x: " ".join(x)
)
subreddit_text = (
    comments_data.groupby("subreddit")["comment_text"]
    .apply(lambda x: " ".join(x))
    .tolist()
)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(subreddit_text)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.index = comments_data.groupby("subreddit").groups.keys()

top_words_dict = {}

for subreddit in tfidf_df.index:
    top_words = tfidf_df.loc[subreddit].sort_values(ascending=False).head(10)
    top_words_dict[subreddit] = top_words.index.tolist()

top_words_df = pd.DataFrame(top_words_dict)
top_words_df["Rank"] = range(1, 11)
cols = ["Rank"] + list(top_words_df.columns[:-1])
top_words_df = top_words_df[cols]
top_words_df.to_csv("results/top_words_per_subreddit.csv", index=False)
