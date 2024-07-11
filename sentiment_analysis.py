from transformers import pipeline
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

comments_data = pd.read_csv("data/comments_cleaned.csv")
comments = comments_data['comment'].astype(str).tolist()

# Load the sentiment-analysis model
model_path ="SamLowe/roberta-base-go_emotions"
pipe = pipeline(task="text-classification", model=model_path, top_k=None,truncation=True)

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
for i,out in enumerate(outputs):
  scores.append(out[0]['score'])
  sentiments.append(out[0]['label'])

comments_data['score'] = scores
comments_data['sentiment'] = sentiments
comments_data.to_csv('results/comments_with_sentiment.csv', index=False)

# Plot the counts of top 1 sentiments
sentiment_count= comments_data.groupby('sentiment').count().reset_index()
sentiment_count.sort_values(by='comment', ascending=True, inplace=True)
sentiment_count = sentiment_count[['sentiment','comment']]
sentiment_count.plot(kind='barh', x='sentiment', y='comment', legend=False)
plt.xlabel('Count')
plt.ylabel('Sentiment')
plt.title('Sentiment Distribution')
plt.savefig('results/sentiment_distribution.png')

# Calculate the proportion of sentiments per post
sentiment_counts_post = comments_data.groupby('post_id')['sentiment'].value_counts().unstack().fillna(0)
post_data = pd.read_csv('posts_cleaned.csv')
post_data = post_data.merge(sentiment_counts_post, on='post_id', how='left')
post_data.iloc[:,7:] = post_data.iloc[:,7:].div(post_data['num_comments'], axis=0)*100
post_data.to_csv('results/posts_with_sentiment.csv', index=False)

# Top 3 posts with the highest proportion of each sentiment
voc = {}
for sentiment in post_data.columns[7:]:
    post_data = post_data.sort_values(by=sentiment, ascending=False)
    voc[sentiment] = post_data['post_id'][0:3].astype(str).tolist()
print(voc)
