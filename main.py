from src.reader import read_essay
from src.statistics import count_characters, count_words, count_sentences

text = read_essay("essays/sample.txt")

characters = count_characters(text)
words = count_words(text)
sentences = count_sentences(text)

print("Essay analysis")
print("Characters:", characters)
print("Words:", words)
print("Sentences:", sentences)