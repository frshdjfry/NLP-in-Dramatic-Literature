import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize

def load_text(filename):
    """Load and return the text from a specified file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

def initialize_pipeline(model_name="tblard/tf-allocine"):
    """Initialize and return the sentiment analysis pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def calculate_valence(sentiment_scores):
    """Calculate and return the valence score for a given list of sentiment scores."""
    return sum(result['score'] if result['label'] == 'POSITIVE' else -result['score'] for result in sentiment_scores)

def segment_text(text, words_per_segment=150):
    """Segment the text into chunks of a specified number of words."""
    text = text.replace('\n', ' ')
    sentences = [s for pre_s in sent_tokenize(text, language='french')
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
    """Analyze the sentiment for each segment and return a list of valence scores."""
    return [sum(calculate_valence(sentiment_pipeline(sentence)) for sentence in segment) / len(segment) for segment in segments]

def moving_average(data, window_size=10):
    """Return the moving average of the given data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def plot_sentiments(sentiments, window_size=10):
    """Plot sentiment intensity with a moving average over time."""
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    palette = plt.get_cmap('Set1')

    smoothed_scores = moving_average(sentiments, window_size)

    plt.plot(sentiments, linestyle='-', alpha=0.3, lw=1.5, label='Original Sentiment Scores', color=palette(0))
    plt.plot(smoothed_scores, linestyle='-', lw=2, label='Smoothed Sentiment (Moving Average)', color=palette(1))

    plt.title('Sentiment Analysis Over Play Segments', loc='left', fontsize=12, color='orange')
    plt.xlabel('Segment Number')
    plt.ylabel('Sentiment Intensity (Positive to Negative)')
    plt.legend(loc='upper left', frameon=True, framealpha=0.7, facecolor='white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()

def main(filename):
    """Main function to process the text and perform sentiment analysis."""
    # Load text
    text = load_text(filename)

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = initialize_pipeline()

    # Segment text and analyze sentiment
    segments = segment_text(text)
    sentiments = analyze_sentiment(segments, sentiment_pipeline)

    # Plot sentiment intensity over segments
    plot_sentiments(sentiments)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    main(sys.argv[1])
