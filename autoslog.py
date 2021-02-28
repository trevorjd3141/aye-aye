import spacy
import util
nlp = spacy.load("en_core_web_sm")

def extractPatterns(text, lexicon, size=2):
    doc = nlp(text)
    patterns = set()
    for sent in doc.sents:
        words = list([word.text for word in sent])
        matches = [(words.index(word), word) for word in words if word in lexicon]
        for index, word in matches:
            windowRight = min(len(sent), index + size + 1)
            windowLeft = max(0, index - size)
            window = sent[windowLeft:windowRight]
            pattern = []
            for token in window:
                if token.text == word:
                    pattern.append(('_WORD_', token.dep_))
                else:
                    pattern.append((token.text, token.dep_))
            patterns.add(tuple(pattern))
    return patterns