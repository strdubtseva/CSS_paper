from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
from data_cleaner import clean_text


def topic_modeling(corpus, num_topics=10, passes=10, random_state=123):
    """
    Function to perform topic modeling using LDA
    Args:
        corpus (list): List of posts' comments
        num_topics (int): Number of topics to extract
        passes (int): Number of passes through the corpus during training
        random_state (int): Random state for reproducibility
    Returns:
        lda_model (LdaModel): Trained LDA model
        corpus_lda (list): List of topics for each post
        coherence_score (float): Coherence score of the model
    """
    # Create a dictionary from the corpus
    dictionary = corpora.Dictionary(corpus)
    # Create a bag-of-words corpus
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
    # Train the LDA model
    lda_model = LdaModel(
        corpus_bow,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=random_state,
    )
    # Get the topics for each document
    corpus_lda = [lda_model[doc] for doc in corpus_bow]
    # Compute the coherence score per topic
    coherence_model = CoherenceModel(
        model=lda_model, texts=corpus, dictionary=dictionary, coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()

    return lda_model, corpus_lda, coherence_score


def find_best_model(corpus, topic_range, passes_range, random_state=123):
    """
    Function to find the best LDA model based on coherence score
    Args:
        corpus (list): List of documents
        topic_range (list): List of topic numbers to try
        passes_range (list): List of passes numbers to try
        random_state (int): Random state for reproducibility
    Returns:
        best_model (LdaModel): Best LDA model based on coherence score
        best_coherence (float): Best coherence score
        best_num_topics (int): Number of topics for the best model
        best_passes (int): Number of passes for the best model
    """
    best_model = None
    best_coherence = -1
    best_num_topics = None
    best_passes = None

    for num_topics in topic_range:
        for passes in passes_range:
            lda_model, _, coherence_score = topic_modeling(
                corpus, num_topics=num_topics, passes=passes, random_state=random_state
            )
            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_model = lda_model
                best_num_topics = num_topics
                best_passes = passes

    return best_model, best_coherence, best_num_topics, best_passes


def top_words_per_topic(lda_model, num_words=20, filename="results/LDA_topics.txt"):
    """
    Function to save the top words per topic to a file
    Args:
        lda_model (LdaModel): Trained LDA model
        num_words (int): Number of top words to save per topic
        filename (str): Filename to save the top words
    """
    num_topics = lda_model.num_topics
    with open(filename, "w") as file:
        for idx, topic in lda_model.show_topics(
            num_topics=num_topics, formatted=False, num_words=num_words
        ):
            file.write(f"Topic {idx+1}:\n")
            words = [word for word, _ in topic]
            file.write(" ".join(words) + "\n\n")
    print(f"Top words per topic saved to {filename}")


if __name__ == "__main__":

    comments_data = pd.read_csv("data/comments_cleaned.csv")
    comments_data["cleaned_comment"] = comments_data["comment"].apply(clean_text)

    comments_data["comment_text"] = comments_data["cleaned_comment"].apply(lambda x: " ".join(x))
    comments_joined = (
        comments_data.groupby("post_id")["comment_text"]
        .apply(lambda x: " ".join(x))
        .tolist()
    )
    corpus = [[word for word in document.lower().split()] for document in comments_joined]

    # Find the best LDA model
    topic_range = [5, 6, 7, 8, 9, 10]
    passes_range = [20, 25, 30, 35, 40]
    best_model, best_coherence, best_num_topics, best_passes = find_best_model(corpus, topic_range, passes_range)

    print(f"Best LDA model: Num topics: {best_num_topics}, Passes: {best_passes}, Coherence score: {best_coherence}")

    # Save the top words per topic
    top_words_per_topic(best_model, num_words=40)
