import pandas as pd
import math
import spacy
from spacy.matcher import DependencyMatcher
nlp = spacy.load("en_core_web_md")
matcher = DependencyMatcher(nlp.vocab)
from datetime import datetime
from collections import defaultdict
import pickle
from os.path import isfile
from itertools import chain

import util
import pattern_generation

PATTERN_POOL_INIT_SIZE = 20
WORDS_PER_ROUND = 5
LOOPS = 15
MAX_TEXT_SIZE = 2000

# Options for filtering what patterns autoslog will return
LEFT_TOKENS=2
PARENT_TOKENS=2
RIGHT_TOKENS=0
MIN_PATTERN_COMPLEXITY=1
MAX_PATTERN_COMPLEXITY=4

def avg_log(patterns, category, lexicon):
    # Creates a list of lists containing only category members
    avg_log_members_extracted = 0
    for pattern in patterns:
        category_words = [word for word in pattern[1] if word in lexicon]
        avg_log_members_extracted += math.log(len(category_words)+1, 2)
    p = len(patterns)

    score = avg_log_members_extracted/p
    return round(score, 3)

def r_log_f(words, category, lexicon):
    n = len(words)
    f = len([word for word in words if word in lexicon])

    if n == 0 or f == 0:
        score = 0
    else:
        score = (f/n)*math.log(f,2)

    return round(score, 3)

def convert_to_dependency_pattern(original_pattern):
    target, lefts, parents, rights = original_pattern
    target_text, target_dep = target
    target_id = f'{target_text}-{target_dep}'
    dependency_pattern = [{'RIGHT_ID': target_id, 'RIGHT_ATTRS': {'DEP': target_dep}}]
    for unit in chain(lefts, rights):
        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': target_id,
            'REL_OP': '>',
            'RIGHT_ID': f'{unit_text}-{unit_dep}',
            'RIGHT_ATTRS': {'ORTH': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)

    for i, unit in enumerate(parents):
        if i == 0:
            left_id = target_id
        else:
            prev_unit = parents[i-1]
            prev_text, prev_dep = prev_unit
            left_id = f'{prev_text}-{prev_dep}'

        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': left_id,
            'REL_OP': '<',
            'RIGHT_ID': f'{unit_text}-{unit_dep}',
            'RIGHT_ATTRS': {'ORTH': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
    return dependency_pattern

def extract(doc, pattern):
    dependency_pattern = convert_to_dependency_pattern(pattern)
    matcher.add('PATTERN', [dependency_pattern])
    matches = matcher(doc)
    matcher.remove('PATTERN')

    extractions = set()
    for match in matches:
        match_id, token_indices = match
        target_index = token_indices[0]
        target_text = doc[target_index].text
        extractions.add(target_text)

    return extractions

def aye_aye(category, output, path, pickle_path, docs_path, development=False):
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))
    seeds = util.fetch_lines(f'seeds/{category}.seed')
    lexicon = set(seeds)

    # List in order to measure precision later on
    generated_words = []

    # Check to see if NLP has been done already
    if isfile(docs_path):
        print('Importing Previously Parsed Documents')
        with open(docs_path, 'rb') as file: 
            docs = pickle.load(file)
    else:
        print('Reading CSV')
        texts = pd.read_csv(path, squeeze=True)
        texts.dropna(inplace=True)
        # Split sampledTexts into series of size <= MAX_TEXT_SIZE and finalize them into a list of docs
        split_texts = np.split(texts, [MAX_TEXT_SIZE*(i+1) for i in range(len(texts)//MAX_TEXT_SIZE)])
        joined_texts = ['. '.join(text) for text in split_texts]
        print('Performing Dependency Parsing')
        docs = [nlp(text) for text in joined_texts]
        with open(docs_path, 'wb') as file:
            pickle.dump(docs, file)
    
    # Check for already extraced patterns
    if isfile(pickle_path):
        with open(pickle_path, 'rb') as file: 
            extracted_patterns_dict = pickle.load(file) 
    else:
        extracted_patterns_dict = defaultdict(set)

    if development:
        print('Pre Processing Done')

    for iteration in range(LOOPS):
        
        if development:
            print(f'Starting Loop {iteration+1} for category {category}'), 

        all_patterns = set()
        for doc in docs:
            all_patterns.update(pattern_generation.extract_patterns(doc, lexicon, LEFT_TOKENS, PARENT_TOKENS, RIGHT_TOKENS, MIN_PATTERN_COMPLEXITY, MAX_PATTERN_COMPLEXITY))

        if development:
            print('1: Patterns Extracted')
            progress = 0
            print(f'Total Patterns for Round {iteration+1}: {len(all_patterns)}')
            
        extracted_patterns = []
        for pattern in all_patterns:
            if pattern in extracted_patterns_dict:
                extracted_patterns.append((pattern, extracted_patterns_dict[pattern]))
            else:
                extracted_pattern = set()
                for doc in docs:
                    extracted_pattern.update(extract(doc, pattern))
                extracted_patterns_dict[pattern] = extracted_pattern
                extracted_patterns.append((pattern, extracted_pattern))
            if development:
                if progress % 500 == 0:
                    print(f'{round((progress/len(all_patterns)), 2)}% of the way finished extracting round {iteration+1}')
                progress += 1

        if development:
            print('2: Extraction Done on All Potential Patterns')

        scored_patterns = []
        for pattern_set in extracted_patterns:
            score = r_log_f(pattern_set[1], category, lexicon)
            scored_patterns.append((pattern_set[0], pattern_set[1], score))
        scored_patterns.sort(key=lambda x: x[2], reverse=True)

        if development:
            print('3: Patterns Scored and Trimmed')

        chosen_patterns = scored_patterns[:PATTERN_POOL_INIT_SIZE + iteration]
        candidate_words = set().union(*[chosen_pattern[1] for chosen_pattern in chosen_patterns])
        candidate_words = candidate_words.difference(lexicon)

        scored_words = []
        for word in candidate_words:
            candidate_word_patterns = [pattern for pattern in extracted_patterns if word in pattern[1]]
            score = avg_log(candidate_word_patterns, category, lexicon)
            scored_words.append((word, score))
        scored_words.sort(key=lambda x: x[1], reverse=True)
        chosen_words = [word[0] for word in scored_words[:WORDS_PER_ROUND]]
        generated_words += chosen_words
        lexicon = lexicon.union(set(chosen_words))

        if development:
            print('4: Words Scored and Trimmed')
            print()

    print('Extracted Words...')
    print(lexicon.difference(set(seeds)))
    print()

    # Write the output of generated words
    file = open(f'{output}{category}.txt','w')
    for word in generated_words:
        file.write(f'{word}\n')
    file.close()

    # Save pattern extractions for next time
    with open(pickle_path, 'wb') as file: 
        pickle.dump(extracted_patterns_dict, file) 
    print("End Time:", datetime.now().strftime("%H:%M:%S"))