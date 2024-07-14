# Computational Social Science Research
This repository contains the code and data for the research paper "Learning or Cheating? Reddit Insights on ChatGPT in Academia". The study analyzes Reddit discussions to uncover themes and sentiments regarding the use of ChatGPT for academic purposes.

## Data Collection
The data is collected from Reddit using the PRAW library. The posts and comments are collected from the following subreddits:
- r/ChatGPT
- r/college
- r/PhD
- r/Professors
- r/ArtificialInteligence
- r/AskAcademia

The search queries are designed to capture discussions about ChatGPT in different subreddits. Titles and texts of posts are searched for the following keywords for each subreddit respectively:
- `university OR college OR exam OR professor`
- `chatgpt`
- `chatgpt`
- `chatgpt`
- `chatgpt AND (university OR college OR exam OR professor)`
- `chatgpt`

The obtained dataset consists of two tables:
- `posts.csv`: Contains data related to individual posts.
- `comments.csv`: Contains data related to comments made on the posts.
Both tables are located in the `data` folder. Additionally, cleaned versions of these datasets are also available in the same folder.

## Data Cleaning and Preprocessing
The collected data is manually filtered to exclude irrelevant posts, and then cleaned to focus on meaningful words.
Data cleaning function includes:
- Removal of special characters
- Conversion of text to lowercase
- Tokenization of text
- Removal of stopwords
- Lemmatization of text

## Analysis
Data analysis is composed of three parts:
- **Exploratory Analysis**: Provides an overview of the data, including the number of posts and comments, generating word clouds to visualize the most frequent words, and calculating TF-IDF metrics to identify the most significant terms in the discussions.
- **Topic Modeling**: Identifies the main topics discussed in the comments using Latent Dirichlet Allocation (LDA) 
- **Sentiment Analysis**: Analyzes the sentiments expressed in the comments to posts using RoBERTa model trained on GoEmotions dataset (https://huggingface.co/SamLowe/roberta-base-go_emotions)

## Results
The `results` folder contains the results of the analysis, including word clouds, TF-IDF table, topics produced by LDA and sentiments extracted from the comments.

## Usage
To run the code follow this pipeline:
1. Install the required packages
2. Run `reddit_data_extractor.py` to collect data (make sure you have Reddit API credentials)
3. Run `data_cleaner.py` to clean and preprocess the data
4. Run `exploratory_analysis.py` for exploratory analysis
5. Run `topic_modeling.py` for topic modeling
6. Run `sentiment_analysis.py` for sentiment analysis

## Requirements
You can install all required packages using the provided `requirements.txt` file.

    ```
    pip install -r requirements.txt
    ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.