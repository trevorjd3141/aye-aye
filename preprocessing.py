from os.path import join
import pandas as pd
import re

CSV_NAMES = ['RAW_recipes.csv', 'RAW_interactions.csv']
CSV_TEXT_COLUMNS = ['description', 'review']

def clean(text):
    text = re.sub('(?:<[^\/]+\/>|&[^;]+;)', '', text)
    return text

def compile(path, name):
    all_text = []
    for i, csv in enumerate(CSV_NAMES):
        text_column = CSV_TEXT_COLUMNS[i]
        df = pd.read_csv(join(path, csv))
        df.dropna(inplace=True)
        for text in df[text_column]:
            text = clean(text)
            all_text.append(text)
    all_text_df = pd.DataFrame({'text': all_text})
    all_text_df.to_csv(join(path, name), index=False)