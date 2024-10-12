import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize


def load_text(filename):
    """Load text from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def initialize_pipeline():
    """Initialize the sentiment analysis pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
    model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def calculate_valence(sentiment_scores):
    """Calculate the valence score for the segment based on sentiment analysis."""
    valence = sum(result['score'] if result['label'] == 'POSITIVE' else -result['score']
                  for result in sentiment_scores)
    return valence


def segment_text(text, words_per_segment=150):
    """Segment the text into chunks of a specified number of words."""
    text = text.replace('\n', ' ')  # Clean text by replacing newlines with spaces
    sentences = [s for pre_s in sent_tokenize(text, language='french')
                 for s in (pre_s.split(',') if len(pre_s) > words_per_segment else [pre_s])]

    segments, current_segment, current_word_count = [], [], 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if words_in_sentence > 1:  # Skip single-word sentences
            if current_word_count + words_in_sentence <= words_per_segment:
                current_segment.append(sentence)
                current_word_count += words_in_sentence
            else:
                segments.append(current_segment)
                current_segment = [sentence]
                current_word_count = words_in_sentence
    if current_segment:
        segments.append(current_segment)  # Add the last segment if not empty

    return segments


def analyze_sentiment(segments, sentiment_pipeline):
    """Analyze the sentiment for each text segment."""
    sentiments = []
    for segment in segments:
        segment_valence = sum(calculate_valence(sentiment_pipeline(sentence))
                              for sentence in segment) / len(segment)
        sentiments.append(segment_valence)
    return sentiments


def moving_average(data, window_size):
    """Calculate the moving average of data using a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_sentiments(sentiments, window_size=10):
    """Plot the sentiment intensity over time with a moving average."""
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    palette = plt.get_cmap('Set1')

    smoothed_scores = moving_average(sentiments, window_size)

    plt.plot(sentiments, marker='', linestyle='-', alpha=0.3, lw=1.5, label='Original Sentiment Scores',
             color=palette(0))
    plt.plot(smoothed_scores, marker='', linestyle='-', lw=2, label='Smoothed Sentiment (Moving Average)',
             color=palette(1))

    plt.title('Sentiment Analysis Over Play Segments', loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel('Segment Number (each representing approx. 1 minute)')
    plt.ylabel('Sentiment Intensity (Positive to Negative)')
    plt.legend(loc='upper left', frameon=True, framealpha=0.7, facecolor='white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    text = load_text(filename)

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = initialize_pipeline()

    # Segment text and analyze sentiment
    segments = segment_text(text)
    sentiments = analyze_sentiment(segments, sentiment_pipeline)

    # Plot sentiment intensity over the segments
    plot_sentiments(sentiments)


if __name__ == "__main__":
    main()
