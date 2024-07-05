import praw
import pandas as pd
import datetime as dt
from prawcore.exceptions import Forbidden, TooManyRequests
import time

start = time.time()
reddit = praw.Reddit(
    client_id='', 
    client_secret='',
    user_agent='', 
    username='',
    password=''
) # Add your credentials

subreddits = ['ChatGPT', 'college', 'PhD', 'Professors', 'ArtificialInteligence', 'AskAcademia']
keywords = [
    'title:university OR title:college OR title:exam OR title:professor OR selftext:university OR selftext:college OR selftext:exam OR selftext:professor',
    'title:chatgpt OR selftext:chatgpt',
    'title:chatgpt OR selftext:chatgpt',
    'title:chatgpt OR selftext:chatgpt',
    '(title:chatgpt OR selftext:chatgpt) AND (title:university OR title:college OR title:exam OR title:professor OR selftext:university OR selftext:college OR selftext:exam OR selftext:professor)',
    'title:chatgpt OR selftext:chatgpt',
]
search_queries = zip(subreddits, keywords)

comments = []
posts = []

def list_moderators(subreddit):
    '''
    Get the list of moderators for a subreddit
    Args: subreddit (str): Name of the subreddit
    Returns: list: List of moderators
    '''
    mods = []
    try:
        for moderator in reddit.subreddit(subreddit).moderator():
            mods.append(moderator.name)  
        mods.append('AutoModerator')
    except Forbidden:
        print(f"Access to the subreddit {subreddit} moderators list is forbidden.")
    return mods

for subreddit, keyword in search_queries:
    moderators = list_moderators(subreddit)
    try:
        for submission in reddit.subreddit(subreddit).search(keyword, sort='top', limit=20):
            submission.comments.replace_more(limit=None)
            top_level_comments = submission.comments
            # Collecting non-deleted, non-removed comments with their scores (score=upvotes-downvotes)
            comments_with_scores = [{
                    'comment_id': comment.id,
                    'comment': comment.body,
                    'score': comment.score,
                    'author': comment.author.name if comment.author else None,
                    'comment_date': comment.created_utc
                } 
                for comment in top_level_comments
                if comment.body not in ['[deleted]', '[removed]']
            ]
            sorted_comments = sorted(comments_with_scores, key=lambda x: x['score'], reverse=True)
            # Add each non-moderator comment as a separate row
            for comment in sorted_comments:
                if comment['author'] not in moderators:
                    comments.append({
                        'post_id': submission.id,
                        'comment_id': comment['comment_id'],
                        'comment': comment['comment'],
                        'comment_score': comment['score'],
                        'comment_date': dt.datetime.fromtimestamp(comment['comment_date']), # Change date type
                    })

            posts.append({
                'post_id': submission.id,
                'post_title': submission.title,
                'post_text': submission.selftext,
                'post_upvotes': submission.score,
                'num_comments': submission.num_comments,
                'post_date': dt.datetime.fromtimestamp(submission.created_utc), # Change date type,
                'subreddit': subreddit,
            })
            time.sleep(2)  # To avoid rate limits
    except TooManyRequests:
        print("Rate limit exceeded. Waiting for a while before retrying.")
        time.sleep(60) 
    except Exception as e:
        print(f"An error occurred: {e}")

comments_data = pd.DataFrame(comments)
comments_data.to_csv('data/comments.csv', index=False)

posts_data = pd.DataFrame(posts)
posts_data.to_csv('data/posts.csv', index=False)

end = time.time()
print(f'Time taken: {end - start} seconds')
print('Data extraction complete!')








