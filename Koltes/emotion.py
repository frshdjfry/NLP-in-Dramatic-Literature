import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Initialize the emotion detection pipeline
emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion',
                            return_all_scores=True)


def load_text(filename):
    """Load text from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def segment_text(text, words_per_segment=150):
    """Segment the text into blocks of a specified number of words."""
    text = text.replace('\n', ' ')  # Clean text by replacing newlines with spaces
    sentences = [s for pre_s in sent_tokenize(text, language='english') for s in
                 (pre_s.split(',') if len(pre_s) > words_per_segment else [pre_s])]

    segments, current_segment, current_word_count = [], [], 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if words_in_sentence > 1:  # Avoid single-word sentences
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


def calculate_average_emotions(nested_list):
    """Calculate the average emotions from a list of nested emotion scores."""
    emotion_sums = {emotion: 0 for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']}
    count = sum(len(inner_list) for outer_list in nested_list for inner_list in outer_list)

    for outer_list in nested_list:
        for inner_list in outer_list:
            for emotion_dict in inner_list:
                emotion_sums[emotion_dict['label']] += emotion_dict['score']

    return {emotion: total / count for emotion, total in emotion_sums.items()}


def analyze_emotions(segments):
    """Analyze the emotions for each segment of the text."""
    return [calculate_average_emotions([emotion_pipeline(sentence) for sentence in segment]) for segment in segments]


def moving_average(data, window_size):
    """Calculate the moving average of a list."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_emotions(emotion_data, window_size=10):
    """Plot the emotions over time with a moving average."""
    emotions_to_plot = ['sadness', 'love', 'fear', 'surprise']
    plt.figure(figsize=(14, 8))
    palette = plt.get_cmap('Set1')

    for i, emotion in enumerate(emotions_to_plot):
        scores = [entry[emotion] for entry in emotion_data]
        smoothed_scores = moving_average(scores, window_size)
        plt.plot(scores, marker='', linestyle='-', alpha=0.3, lw=1.5, label=f'Original {emotion.title()}',
                 color=palette(i))
        plt.plot(smoothed_scores, marker='', lw=2, label=f'Smoothed {emotion.title()} (Moving Average)',
                 color=palette(i), linestyle='--')

    plt.title('Emotion Intensity Over Time', loc='left', fontsize=14, fontweight=0, color='orange')
    plt.xlabel('Segment Number')
    plt.ylabel('Emotion Intensity')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, framealpha=0.7, facecolor='white')
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_emotion_percentages(emotion_data):
    """Plot the percentage of each emotion as a pie chart."""
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    emotion_sums = {emotion: sum(entry[emotion] for entry in emotion_data) for emotion in emotions}

    total_sum = sum(emotion_sums.values())
    emotion_percentages = {emotion: (score / total_sum) * 100 for emotion, score in emotion_sums.items()}

    plt.figure(figsize=(8, 8))
    plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(), autopct='%1.1f%%', startangle=140,
            colors=plt.get_cmap('Set3')(np.linspace(0, 1, len(emotions))))
    plt.axis('equal')
    plt.title('Percentage of Each Emotion in Total')
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    text = load_text(filename)

    # Segment and analyze text
    all_segments = segment_text(text)
    emotions = analyze_emotions(all_segments)

    # Plot the results
    plot_emotions(emotions)
    plot_emotion_percentages(emotions)


if __name__ == "__main__":
    main()
