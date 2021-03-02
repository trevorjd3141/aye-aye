import spacy
import util
nlp = spacy.load("en_core_web_md")

def extractPatterns(doc, lexicon, MIN_WINDOW_SIZE=3, MAX_WINDOW_SIZE=6):
    patterns = set()
    for sent in doc.sents:
        words = [word.text for word in sent]
        matches = [(words.index(word), word) for word in words if word in lexicon]
        for index, word in matches:
            fullSentencePattern = []
            for token in sent:
                if token.text == word:
                    fullSentencePattern.append(('_WORD_', token.dep_))
                else:
                    fullSentencePattern.append((token.text, token.dep_))
            allExtractedPatterns = []
            for i in range(index):
                for j in range(index, len(sent) + 1):
                    allExtractedPatterns.append(fullSentencePattern[i:j+1])
            allExtractedPatterns = [pattern for pattern in allExtractedPatterns if len(pattern) >= MIN_WINDOW_SIZE and len(pattern) <= MAX_WINDOW_SIZE]
            for pattern in allExtractedPatterns:
                patterns.add(tuple(pattern))
    return patterns