import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy

def validate_args():
    """Ensure a filename is provided as a command-line argument."""
    if len(sys.argv) < 2:
        print("Usage: python script.py filename.txt")
        sys.exit(1)
    return sys.argv[1]

def load_text(file_path):
    """Load and return the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

def process_text(text, nlp):
    """Process the text using spaCy to filter tokens and return lemmatized words."""
    doc = nlp(text)
    words = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.pos_ != "VERB"
    ]
    return words

def generate_wordcloud(words):
    """Generate and display a word cloud from the list of words."""
    wordcloud = WordCloud(background_color="white", width=800, height=400, collocations=False).generate(' '.join(words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def calculate_word_frequency(words):
    """Calculate and return word frequencies as a dictionary."""
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

def calculate_vocabulary_richness(words):
    """Calculate and return the type-token ratio (vocabulary richness)."""
    types = len(set(words))
    tokens = len(words)
    return types / tokens if tokens > 0 else 0

def print_most_common_words(word_freq, limit=100):
    """Print the most common words from the word frequency dictionary."""
    print('\nMost common words:')
    for word, count in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:limit]:
        print(f'{word}: {count}')

def main():
    # Validate arguments and load the text
    filename = validate_args()
    text = load_text(filename)

    # Load the language model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    words = process_text(text, nlp)

    # Generate word cloud
    generate_wordcloud(words)

    # Calculate word frequency and vocabulary richness
    word_freq = calculate_word_frequency(words)
    v_richness = calculate_vocabulary_richness(words)

    # Output results
    print(f'Vocabulary Richness (Type-Token Ratio): {v_richness:.4f}')
    print_most_common_words(word_freq)

if __name__ == "__main__":
    main()
