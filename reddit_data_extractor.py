import praw
import pandas as pd
import datetime as dt

reddit = praw.Reddit(
    client_id='', 
    client_secret='',
    user_agent='reddit_data_extractor', 
    username='',
    password='')  #TODO DELETE CLIENT ID and SECRET, USERNAME, PASSWORD


subreddit = ['ChatGPT', 'college']
keywords = ['university AND cheating','chatGPT']
search_queries = zip(subreddit, keywords)

comments = []
posts = []

def list_moderators(subreddit):
    mods = []
    for moderator in reddit.subreddit(subreddit).moderator():
        mods.append(moderator)
    mods.append('AutoModerator')
    return mods

for subreddit, keyword in search_queries:
    for submission in reddit.subreddit(subreddit).search(keyword, sort='relevance', limit=1):
        # Including only comments to posts (no replies to comments)
        submission.comments.replace_more(limit=None)
        top_level_comments = submission.comments
        # Collecting non-deleted, non-removed comments with their scores (score=upvotes-downvotes)
        comments_with_scores = [{
                'comment_id': comment.id,
                'comment': comment.body,
                'score': comment.score,
                'author': comment.author,
                'comment_date': comment.created_utc
            } 
            for comment in top_level_comments
            if comment.body not in ['[deleted]', '[removed]']
        ]
        # Sorting comments by score in descending order
        sorted_comments = sorted(comments_with_scores, key=lambda x: x['score'], reverse=True)
        # Add each non-moderator comment as a separate row
        for comment in sorted_comments:
            if comment['author'] not in list_moderators(subreddit):
                comments.append({
                    'post_id': submission.id,
                    'comment_id': comment['comment_id'],
                    'comment': comment['comment'],
                    'comment_score': comment['score'],
                    'comment_date': dt.datetime.fromtimestamp(comment['comment_date']), # Change date type
                })
        # Add post details
        posts.append({
            'post_id': submission.id,
            'post_title': submission.title,
            'post_text': submission.selftext,
            'post_upvotes': submission.score,
            'num_comments': submission.num_comments,
            'post_date': dt.datetime.fromtimestamp(submission.created_utc), # Change date type
        })

comments_data=pd.DataFrame(comments)
comments_data.to_csv('comments.csv', index=False)

posts_data=pd.DataFrame(posts)
posts_data.to_csv('posts.csv', index=False)








