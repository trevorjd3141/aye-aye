import spacy
import util
nlp = spacy.load("en_core_web_md")

def extractPatterns(doc, lexicon):
    patterns = set()
    for sent in doc.sents:
        words = [word.text for word in sent]
        matches = [(words.index(word.text), word) for word in sent if word.dep_ != 'ROOT' and word.text in lexicon]
        for index, token in matches:
            patterns.add((token.head.text, token.dep_))
    return patterns