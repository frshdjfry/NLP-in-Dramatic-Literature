import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize


def validate_args():
    """Ensure a filename is provided as a command-line argument."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    return sys.argv[1]


def initialize_sentiment_pipeline():
    """Initialize the sentiment analysis pipeline with a specified model."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def load_text(file_path):
    """Load and return text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)


def calculate_valence(sentiment_scores):
    """Calculate the valence score for the sentiment of a sentence."""
    valence = sum(result['score'] if result['label'] == 'POSITIVE' else -result['score'] for result in sentiment_scores)
    return valence


def segment_text(text, words_per_segment=15):
    """Segment the text into chunks of a specified number of words."""
    text = text.replace('\n', ' ')
    sentences = [s for pre_s in sent_tokenize(text, language='english')
                 for s in (pre_s.split(',') if len(pre_s) > words_per_segment else [pre_s])]

    segments, current_segment, current_word_count = [], [], 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if words_in_sentence > 1:
            if current_word_count + words_in_sentence <= words_per_segment:
                current_segment.append(sentence)
                current_word_count += words_in_sentence
            else:
                segments.append(current_segment)
                current_segment = [sentence]
                current_word_count = words_in_sentence
    if current_segment:
        segments.append(current_segment)

    return segments


def analyze_sentiment(segments, sentiment_pipeline):
    """Analyze the sentiment of each segment and return a list of valence scores."""
    sentiments = []
    for segment in segments:
        segment_sentiments = [calculate_valence(sentiment_pipeline(sentence)) for sentence in segment]
        avg_segment_sentiment = sum(segment_sentiments) / len(segment_sentiments)
        sentiments.append(avg_segment_sentiment)
    return sentiments


def moving_average(data, window_size):
    """Calculate and return the moving average using a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_sentiments(sentiments, window_size=10):
    """Plot sentiment intensity with a moving average over time."""
    smoothed_scores = moving_average(sentiments, window_size)

    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    palette = plt.get_cmap('Set1')

    plt.plot(smoothed_scores, linestyle='-', lw=2, label='Smoothed Sentiment (Moving Average)', color=palette(1))

    plt.title('Sentiment Analysis Over Play Segments', loc='left', fontsize=12, color='orange')
    plt.xlabel('Segment Number')
    plt.ylabel('Sentiment Intensity (Positive to Negative)')
    plt.legend(loc='upper left', frameon=True, framealpha=0.7, facecolor='white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()


def main():
    filename = validate_args()
    text = load_text(filename)

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = initialize_sentiment_pipeline()

    # Segment text and analyze sentiment
    segments = segment_text(text)
    sentiments = analyze_sentiment(segments, sentiment_pipeline)

    # Plot sentiment intensity over the segments
    plot_sentiments(sentiments)


if __name__ == "__main__":
    main()
