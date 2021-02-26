import spacy
import util
nlp = spacy.load("en_core_web_sm")

def autoslogTS(path=None, lexicon=None, size=2):
    lexicon = ['Apple', 'Microsoft']
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion. I would like to run in the dark. Microsoft is based in Redmond.")
    patterns = set()
    for sent in doc.sents:
        words = list([word.text for word in sent])
        matches = [(words.index(word), word) for word in words if word in lexicon]
        print(matches)
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
    print(patterns)

autoslogTS()