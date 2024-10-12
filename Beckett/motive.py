import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict


def preprocess_text(text):
    """Convert text to lowercase, remove punctuation and special characters, and clean up spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())  # Remove extra spaces


def longest_repeated_substrings(text, max_length=260):
    """Find the longest repeated substrings in the text."""
    n = len(text)
    suffixes = sorted([text[i:] for i in range(n)])
    lcp = [0] * n  # Longest common prefix array

    def lcp_length(s1, s2):
        length = 0
        while length < len(s1) and length < len(s2) and s1[length] == s2[length]:
            length += 1
        return length

    for i in range(1, n):
        lcp[i] = lcp_length(suffixes[i - 1], suffixes[i])

    substrings = defaultdict(list)
    for i in range(1, n):
        if lcp[i] > 0:
            substring = suffixes[i][:lcp[i]]
            substrings[substring].append(i - lcp[i])

    filtered_substrings = [sub for sub in substrings if len(sub) <= max_length]
    filtered_substrings.sort(key=len, reverse=True)

    return filtered_substrings[:10]


def find_substring_positions(text, substring):
    """Find all positions of a substring in the text."""
    positions = []
    pos = text.find(substring)
    while pos != -1:
        positions.append(pos)
        pos = text.find(substring, pos + 1)
    return positions


def filter_subsequences(subsequences):
    """Filter out subsequences that are contained within longer subsequences."""
    filtered = []
    for sub in subsequences:
        if not any(sub in longer_sub for longer_sub in filtered):
            filtered.append(sub)
    return filtered


def prepare_gantt_data(text):
    """Prepare Gantt chart data by finding longest repeated substrings and their positions."""
    substrings = longest_repeated_substrings(text)
    filtered_substrings = filter_subsequences(substrings)
    data = []
    for substring in filtered_substrings:
        positions = find_substring_positions(text, substring)
        for pos in positions:
            data.append((substring, pos, pos + len(substring)))
    return data, filtered_substrings


def plot_gantt_chart(data, unique_substrings):
    """Plot a Gantt chart of the longest repeated substrings in the text."""
    fig, ax = plt.subplots()

    for i, substring in enumerate(unique_substrings):
        positions = [(start, end - start) for sub, start, end in data if sub == substring]
        ax.broken_barh(positions, (i - 0.4, 0.8), facecolors=('tab:blue'))
        for start, _ in positions:
            ax.text(start, i, '', ha='left', va='center', fontsize=10, color='black')

    ax.set_yticks(range(len(unique_substrings)))
    ax.set_yticklabels([f'{substring[:10]}...{substring[-10:]}' for substring in unique_substrings])
    ax.set_xlabel('Position in Text (characters)')
    ax.set_ylabel('Motifs')
    ax.set_title('Longest Repetitive Subsequences (Motifs) in Text')
    plt.show()


def main():
    """Main function to load text, process it, and generate a Gantt chart."""
    with open('full_text.txt', 'r') as file:
        raw_text = file.read()

    # Preprocess the text
    text = preprocess_text(raw_text)

    # Prepare data and plot the Gantt chart
    data, unique_substrings = prepare_gantt_data(text)
    plot_gantt_chart(data, unique_substrings)


if __name__ == "__main__":
    main()
