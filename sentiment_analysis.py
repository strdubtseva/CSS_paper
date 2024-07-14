from transformers import pipeline
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json


comments_data = pd.read_csv("data/comments_cleaned.csv")
comments = comments_data['comment'].astype(str).tolist()

# Load the sentiment-analysis model
model_path = "SamLowe/roberta-base-go_emotions"
pipe = pipeline(task="text-classification", model=model_path, top_k=None, truncation=True)


# Create a dataset class
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


textdataset = TextDataset(comments)
outputs = pipe(textdataset)

# Add the top 1 score and sentiment to comments
scores = []
sentiments = []
for i, out in enumerate(outputs):
    scores.append(out[0]['score'])
    sentiments.append(out[0]['label'])

comments_data['score'] = scores
comments_data['sentiment'] = sentiments
comments_data.to_csv('results/comments_with_sentiment.csv', index=False)

posts_data = pd.read_csv("data/posts_cleaned.csv")
comments_data = comments_data.merge(posts_data[["post_id", "subreddit"]], on="post_id")

# Get proportions of sentiments per post
sentiment_counts_post = comments_data.groupby('post_id')['sentiment'].value_counts().unstack().fillna(0)
posts_data = posts_data.merge(sentiment_counts_post, on='post_id', how='left')
posts_data.iloc[:, 7:] = posts_data.iloc[:, 7:].div(posts_data['num_comments'], axis=0)*100
posts_data.to_csv('results/posts_with_sentiment.csv', index=False)

# Get top 3 posts for each sentiment
voc = {}
for sentiment in posts_data.columns[7:]:
    posts_data = posts_data.sort_values(by=sentiment, ascending=False)
    voc[sentiment] = posts_data['post_id'][0:3].astype(str).tolist()

# Save top 3 posts for each sentiment as a json file
js = json.dumps(voc)
fp = open('results/top3_posts_by_sentiment', 'a')
fp.write(js)
fp.close()

# Plot the percentages of comments per sentiment
comments_data = comments_data[comments_data['sentiment'] != 'neutral']
sentiment_count = comments_data.groupby('sentiment').count().reset_index()
sentiment_count.sort_values(by='comment', ascending=True, inplace=True)
sentiment_count = sentiment_count[['sentiment', 'comment']]
col = plt.colormaps['tab20c'].colors[1]
sentiment_count['comment'] = sentiment_count['comment']/sentiment_count['comment'].sum()*100.0
fig = plt.figure(figsize=(9, 11))
ax = sentiment_count.plot(kind='barh', x='sentiment', y='comment', legend=False, color=col)
plt.xlabel('Percentage of comments')
plt.xlim(0, 15)
ax.bar_label(ax.containers[0], fmt='{:,.1f}', color=col, padding=8)
plt.ylabel('Sentiment')
plt.savefig('results/sentiment_percentage.png', bbox_inches='tight')

# Plot the percentages of likes per sentiment
comments_grouped = comments_data.groupby('sentiment')['comment_score'].sum()
comments_grouped = comments_grouped.reset_index()
comments_grouped['comment_score'] = comments_grouped['comment_score']/comments_grouped['comment_score'].sum()*100.0
comments_grouped.sort_values(by='comment_score', ascending=True, inplace=True)
ax = comments_grouped.plot(kind='barh', x='sentiment', y='comment_score', legend=False, color=col)
ax.bar_label(ax.containers[0], fmt='{:,.1f}', color=col, padding=6)
plt.xlim(0, 21)
plt.xlabel('Percentage of comment scores')
plt.ylabel('Sentiment')
plt.savefig('results/sentiment_percentage_likes.png', bbox_inches='tight')

# PLot number of comments with sentiment per subreddit
comments_grouped = comments_data.groupby(['sentiment', 'subreddit'])['comment_score'].sum().unstack()
total_values = comments_grouped.sum(axis=1)
sorted_category1 = total_values.sort_values(ascending=True).index
comments_grouped = comments_grouped.loc[sorted_category1]
ax = comments_grouped.plot(kind='barh', stacked=True)
plt.xlabel('Number of comments')
plt.ylabel('Sentiment')
plt.legend(title='Subreddit')
plt.savefig('results/comments_by_subreddit.png', bbox_inches='tight')
