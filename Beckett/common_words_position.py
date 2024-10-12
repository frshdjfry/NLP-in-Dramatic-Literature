import re
import matplotlib.pyplot as plt
import numpy as np

# List of top words
top_words = """time
buzzing
long
what
like
oh
sudden
flash
mouth
love
morning
far
day
till
brain
thought
hand
sound
word
stream
little
thing
suddenly
april
""".split()


def preprocess_text(text):
    """Preprocess the text by converting to lowercase and removing punctuation/special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation and special characters
    return ' '.join(text.split())  # Clean up any extra spaces


def find_word_positions(text, words):
    """Find the positions of each word in the text."""
    positions = {word: [] for word in words}
    text_words = text.split()
    for i, word in enumerate(text_words):
        if word in positions:
            positions[word].append(i)
    return positions


def plot_word_occurrences(word_positions, words):
    """Plot a Gantt chart showing occurrences of words in the text."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define a color map with professional colors
    colors = plt.cm.get_cmap('Set2', len(words))

    # Plot each word's positions
    for i, word in enumerate(words):
        y = np.full(len(word_positions[word]), i)
        ax.scatter(word_positions[word], y, color=colors(i), s=50, label=word)

    # Customize the plot
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Position in text')
    ax.set_ylabel('Words')
    ax.set_title('Repeated Word Occurrences')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Remove legend for a cleaner look
    ax.legend().set_visible(False)

    # Show the plot
    plt.show()


def main():
    # Load and preprocess the text
    with open('full_text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    text = preprocess_text(text)

    # Find positions of the top words
    word_positions = find_word_positions(text, top_words)

    # Plot the word occurrences
    plot_word_occurrences(word_positions, top_words)


if __name__ == "__main__":
    main()
