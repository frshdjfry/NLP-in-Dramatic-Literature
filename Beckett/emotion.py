import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize


def validate_args():
    """Ensure that a filename is provided as a command-line argument."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    return sys.argv[1]


def load_text(filename):
    """Load and return the text from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)


def initialize_emotion_pipeline():
    """Initialize the emotion classification pipeline."""
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)


def segment_text(text, words_per_segment=30):
    """Segment the text into chunks based on a specified word limit."""
    text = text.replace('\n', ' ')  # Clean text by replacing newlines with spaces
    pre_sentences = sent_tokenize(text, language='english')

    sentences = []
    for s in pre_sentences:
        sentences.extend(s.split(',') if len(s) > words_per_segment else [s])

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


def calculate_average_emotions(nested_list):
    """Calculate the average emotion scores from the nested list of emotions."""
    emotion_sums = {
        'sadness': 0, 'joy': 0, 'love': 0, 'anger': 0,
        'fear': 0, 'disgust': 0, 'surprise': 0, 'neutral': 0
    }
    count = 0

    for outer_list in nested_list:
        for inner_list in outer_list:
            for emotion_dict in inner_list:
                emotion_sums[emotion_dict['label']] += emotion_dict['score']
            count += 1

    return {emotion: total / count for emotion, total in emotion_sums.items()}


def analyze_emotions(segments, emotion_pipeline):
    """Analyze emotions for each segment of text."""
    emotions = []
    for segment in segments:
        sentence_emotions = [emotion_pipeline(sentence) for sentence in segment]
        avg_segment_emotion = calculate_average_emotions(sentence_emotions)
        emotions.append(avg_segment_emotion)
    return emotions


def moving_average(data, window_size=10):
    """Calculate the moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_emotions(emotion_data, window_size=10):
    """Plot the emotion intensities over time with a moving average."""
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    plt.figure(figsize=(14, 8))
    palette = plt.get_cmap('tab10')

    for i, emotion in enumerate(emotions):
        scores = [entry[emotion] for entry in emotion_data]
        smoothed_scores = moving_average(scores, window_size)
        plt.plot(smoothed_scores, lw=2, label=f'{emotion.title()}', color=palette(i), linestyle='--')

    plt.title('Emotion Intensity Over Time (Moving Average)', loc='left', fontsize=14, color='orange')
    plt.xlabel('Segment Number')
    plt.ylabel('Emotion Intensity')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, framealpha=0.7, facecolor='white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_emotion_percentages(emotion_data):
    """Plot a pie chart of emotion percentages."""
    emotions = ['sadness', 'joy', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    emotion_sums = {emotion: sum(entry[emotion] for entry in emotion_data) for emotion in emotions}

    total_sum = sum(emotion_sums.values())
    emotion_percentages = {emotion: (score / total_sum) * 100 for emotion, score in emotion_sums.items()}

    plt.figure(figsize=(8, 8))
    plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(), autopct='%1.1f%%',
            startangle=140, colors=plt.get_cmap('Set3')(np.linspace(0, 1, len(emotions))))
    plt.axis('equal')
    plt.title('Percentage of Each Emotion in Total')
    plt.show()


def main():
    filename = validate_args()
    text = load_text(filename)

    # Initialize the emotion pipeline
    emotion_pipeline = initialize_emotion_pipeline()

    # Segment the text and analyze emotions
    segments = segment_text(text)
    emotions = analyze_emotions(segments, emotion_pipeline)

    # Plot the emotion intensity and percentages
    plot_emotions(emotions)
    plot_emotion_percentages(emotions)


if __name__ == "__main__":
    main()
