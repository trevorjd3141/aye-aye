import util
import autoslog
import pandas as pd
import math
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
nlp = spacy.load("en_core_web_md")

PATTERN_POOL_INIT_SIZE = 20
WORDS_PER_ROUND = 3
LOOPS = 2
MIN_WINDOW_SIZE=3
MAX_WINDOW_SIZE=6
SAMPLE_SIZE = 250
DATA_PATH = 'data\\'

def AvgLog(patterns, category, lexicon):
    # Creates a list of lists containing only category members
    AvgLogMembersExtracted = 0
    for pattern in patterns:
        categoryWords = [word for word in pattern[1] if word in lexicon]
        AvgLogMembersExtracted += math.log(len(categoryWords)+1, 2)
    p = len(patterns)

    score = AvgLogMembersExtracted/p
    return round(score, 3)

def RlogF(words, category, lexicon):
    n = len(words)
    f = len([word for word in words if word in lexicon])

    if n == 0 or f == 0:
        score = 0
    else:
        score = (f/n)*math.log(f,2)

    return round(score, 3)

def extract(text, docPath, pattern):
    patternWords = [token[0] for token in pattern]

    # Exit early if word can't be found
    # in the text
    for word in patternWords:
        if word == '_WORD_':
            continue

        if word not in text:
            return set()

    doc = Doc(Vocab()).from_disk(docPath) 
    size = len(pattern)

    # Index of the word within the pattern
    wordIndex = patternWords.index('_WORD_')

    # The word's dependency
    wordDep = pattern[wordIndex][1]

    # The pattern minus the word
    patternWithoutWord = tuple([pair for pair in pattern if pair[0] != '_WORD_'])

    extractions = set()
    for sent in doc.sents:
        
        # Exit early if all pattern words aren't in the sentence
        sentWords = { word.text for word in sent }
        if not all([word in sentWords for word in patternWords if word != '_WORD_']):
            continue

        for i in range(len(sent) - size + 1):
            window = [(word.text, word.dep_) for word in sent][i: i + size]
            extraction = window[wordIndex]
            extractionText = extraction[0]
            extractionDep = extraction[1]
            del window[wordIndex]
            window = tuple(window)

            if window == patternWithoutWord and extractionDep == wordDep:
                extractions.add(extractionText)
    return extractions

def basilisk(category, output, development=False):
    seeds = util.fetchLines(f'seeds/{category}.seed')
    lexicon = set(seeds)
    textDF = pd.read_csv('data/compiledText.csv')
    textDF = textDF.sample(n=SAMPLE_SIZE)
    textDF.dropna(inplace=True)

    for iteration in range(LOOPS):
        
        if development:
            print(f'Starting Loop {iteration+1} for category {category}'), 
            print('1: Loading Done')

        allPatterns = []
        for file in textDF['File']:
            doc = Doc(Vocab()).from_disk(f'{ANNOTATIONS_PATH}{file}')
            patterns = autoslog.extractPatterns(doc, lexicon, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE)
            allPatterns.append(patterns)
        allPatterns = set().union(*allPatterns)

        if development:
            print('2: Patterns Extracted')

        extractedPatterns = []
        for pattern in allPatterns:
            allExtractedWords = []
            for text, file in zip(textDF['Text'], textDF['File']):
                allExtractedWords.append(extract(text, f'{ANNOTATIONS_PATH}{file}', pattern))
            allExtractedWords = set().union(*allExtractedWords)
            extractedPatterns.append((pattern, allExtractedWords))

        if development:
            print('3: Extraction Done on All Potential Patterns')

        scoredPatterns = []
        for patternSet in extractedPatterns:
            score = RlogF(patternSet[1], category, lexicon)
            scoredPatterns.append((patternSet[0], patternSet[1], score))
        scoredPatterns.sort(key=lambda x: x[2], reverse=True)

        if development:
            print('4: Patterns Scored and Trimmed')

        chosenPatterns = scoredPatterns[:PATTERN_POOL_INIT_SIZE + iteration]
        candidateWords = set().union(*[chosenPattern[1] for chosenPattern in chosenPatterns])
        candidateWords = candidateWords.difference(lexicon)

        scoredWords = []
        for word in candidateWords:
            patternsThatExtracted = [pattern for pattern in extractedPatterns if word in pattern[1]]
            score = AvgLog(patternsThatExtracted, category, lexicon)
            scoredWords.append((word, score))
        scoredWords.sort(key=lambda x: x[1], reverse=True)
        chosenWords = scoredWords[:WORDS_PER_ROUND]
        lexicon = lexicon.union({word[0] for word in chosenWords})

        if development:
            print('5: Words Scored and Trimmed')
            print(chosenWords)
            print()

    print('Extracted Words...')
    generatedWords = lexicon.difference(set(seeds))
    print(lexicon.difference(set(seeds)))
    print()
    file = open(f'{output}{category}.txt','w')
    for word in generatedWords:
        file.write(f'{word}\n')
    file.close() 