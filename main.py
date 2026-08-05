import spacy

nlp = spacy.load("en_core_web_sm")

text = "I am learning computational linguistics."

doc = nlp(text)

for token in doc:
    print(token.text)
    print(token.lemma_)
    print(token.pos_)