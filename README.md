# Computational Social Science Research
This repository contains the code and data for the research paper "Learning or Cheating? Reddit Insights on ChatGPT in Academia". The study analyzes Reddit discussions to uncover themes and sentiments regarding the use of ChatGPT for academic purposes.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Analysis](#analysis)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Introduction
The aim of the research is to discover which themes and sentiments emerge in Reddit discussions about using ChatGPT for academic purposes.

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

## Data Cleaning and Preprocessing
The collected data is manually filtered to exclude irrelevant posts, and then cleaned to focus on meaningful words.

## Analysis
Data analysis is composed of three parts:
- **Descriptive Analysis**: Provides an overview of the data, including the number of posts and comments, generating word clouds to visualize the most frequent words, and calculating TF-IDF metrics to identify the most significant terms in the discussions.
- **Sentiment Analysis**: Analyzes the sentiments expressed in the posts and comments using ////
- **Topic Modeling**: Identifies the main topics discussed in the posts and comments using Latent Dirichlet Allocation (LDA) 

## Usage
To run the code follow this pipeline:
1. Install the required packages
2. Run `reddit_data_extractor.py` to collect data (make sure you have Reddit API credentials)
3. Run `data_cleaner.py` to clean and preprocess the data
4. Run `descriptive_analysis.py` for descriptive analysis
5. Run `sentiment_analysis.py` for sentiment analysis
6. Run `topic_modeling.py` for topic modeling

## Requirements
You can install all required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.