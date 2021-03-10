import util
import pandas as pd
from collections import defaultdict

def compileCommonWords(path, category):
    commonWords = defaultdict(int)
    texts = pd.read_csv(path, squeeze=True)
    for text in texts:
        words = text.split()
        for word in words:
            if util.categorize(word) == category:
                commonWords[word.lower()] += 1
    commonWordsDF = pd.DataFrame.from_dict({'Word': commonWords.keys(), 'Count': commonWords.values()})
    commonWordsDF.sort_values(by='Count', ascending=False, inplace=True)
    commonWordsDF.to_csv(f'seeds/{category}-counts.csv', index=False)