import util
import pandas as pd
from collections import defaultdict

def compile_common_words(path, category):
    common_words = defaultdict(int)
    texts = pd.read_csv(path, squeeze=True)
    for text in texts:
        words = text.split()
        for word in words:
            if util.categorize(word) == category:
                common_words[word.lower()] += 1
    common_words_df = pd.DataFrame.from_dict({'Word': common_words.keys(), 'Count': common_words.values()})
    common_words_df.sort_values(by='Count', ascending=False, inplace=True)
    common_words_df.to_csv(f'seeds/{category}-counts.csv', index=False)