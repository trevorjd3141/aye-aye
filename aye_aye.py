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
import itertools
from operator import itemgetter
import numpy as np
from random import sample
from statistics import mean

import util
import pattern_generation

PATTERN_POOL_INIT_SIZE = 20
PATTERN_POOL_SIZE_INCREASE = 1
WORDS_PER_ROUND = 10
LOOPS = 50
MAX_TEXT_SIZE = 300
MAX_NEW_PATTERNS_PER_ROUND = 2000
# How often should progress be printed?
# 100/PROGRESS_SKIP times
PROGRESS_SKIP = 5

# When filtering based on drift keep the
# top WORDS_PER_ROUND + len(words) * DRIFT_PERCENTAGE words
DRIFT_PERCENTAGE = 0.6

def drift_filter(words, lexicon, count=WORDS_PER_ROUND):
    print('words being drifted', len(words))
    if len(words) <= count*2:
        return words

    firsts = lexicon[:count]
    lasts = lexicon[-count:]
    drift_scores = []
    for word in words:
        token = nlp(word)

        first_similarities = [token.similarity(nlp(first)) for first in firsts]
        first_similarity = mean(first_similarities)

        last_similarities = [token.similarity(nlp(last)) for last in lasts]
        last_similarity = mean(last_similarities)

        drift_score = first_similarity/last_similarity
        drift_scores.append((word, drift_score))
    drift_scores.sort(key=itemgetter(1), reverse=True)
    # for a drift percentage of 0.5 and 10 words per round this would be
    # 10 + half the remaining words. Making sure to remove from the low scorers
    number_kept = WORDS_PER_ROUND + round(DRIFT_PERCENTAGE*(len(words)-WORDS_PER_ROUND))
    filtered_words = [word[0] for word in drift_scores][:number_kept]
    return set(filtered_words)

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
    target_dep, lefts, parents, rights, left_sibling, right_sibling = original_pattern
    target_id = f'target-{target_dep}'
    dependency_pattern = [{'RIGHT_ID': target_id, 'RIGHT_ATTRS': {'DEP': target_dep}}]

    increment = 0
    for unit in itertools.chain(lefts, rights):
        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': target_id,
            'REL_OP': '>',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'LOWER': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1

    for i, unit in enumerate(parents):
        if i == 0:
            left_id = target_id
        else:
            prev_unit = parents[i-1]
            prev_text, prev_dep = prev_unit
            left_id = f'{prev_text}-{prev_dep}-{increment-1}'

        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': left_id,
            'REL_OP': '<',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'LOWER': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1

    if left_sibling:
        # left sibling is a tuple of tuples but really should only have one unit
        # making it a tuple of tuples is to make it consistent with everything else
        # same goes for right sibling
        unit_text, unit_dep = left_sibling[0]
        pattern_unit = {
            'LEFT_ID': target_id,
            'REL_OP': '$-',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'LOWER': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1

    if right_sibling:
        unit_text, unit_dep = right_sibling[0]
        pattern_unit = {
            'LEFT_ID': target_id,
            'REL_OP': '$+',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'LOWER': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1
    return dependency_pattern

def aye_aye(settings, output, path, pickle_path, docs_path, mutual_exclusion, semantic_drift, development=False):
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))

    categories = [category_settings['NAME'] for category_settings in settings]
    seeds = {}
    lexicons = {}
    for category in categories:
        category_seeds = util.fetch_lines(f'seeds/{category}.seed')
        seeds[category] = set(category_seeds)
        lexicons[category] = category_seeds

    # Check to see if NLP has been done already
    if isfile(docs_path):
        if development:
            print('Importing Previously Parsed Documents')
        with open(docs_path, 'rb') as file: 
            docs = pickle.load(file)
    else:
        if development:
            print('Reading CSV')
        texts = pd.read_csv(path, squeeze=True)
        texts.dropna(inplace=True)
        # Split sampledTexts into series of size <= MAX_TEXT_SIZE and finalize them into a list of docs
        split_texts = np.split(texts, [MAX_TEXT_SIZE*(i+1) for i in range(len(texts)//MAX_TEXT_SIZE)])
        joined_texts = ['. '.join(text) for text in split_texts]
        if development:
            print('Performing Dependency Parsing')
        docs = [nlp(text) for text in joined_texts]
        docs = [doc for doc in docs if len(doc) > 0]
        with open(docs_path, 'wb') as file:
            pickle.dump(docs, file)
    
    # Check for already extraced patterns
    if isfile(pickle_path):
        with open(pickle_path, 'rb') as file: 
            extracted_patterns_dict = pickle.load(file)
    else:
        extracted_patterns_dict = defaultdict(set)

    top_patterns = defaultdict(set)

    if development:
        print('Pre Processing Done')

    for iteration in range(LOOPS):
        for category in categories:
            category_lexicon = lexicons[category]
            category_seeds = seeds[category]
            category_settings = next(filter(lambda x: x['NAME'] == category, settings), None)
            left_tokens = category_settings['LEFT_TOKENS']
            parent_tokens = category_settings['PARENT_TOKENS']
            right_tokens = category_settings['RIGHT_TOKENS']
            left_sibling = category_settings['LEFT_SIBLING']
            right_sibling = category_settings['RIGHT_SIBLING']
            min_pattern_complexity = category_settings['MIN_PATTERN_COMPLEXITY']
            max_pattern_complexity = category_settings['MAX_PATTERN_COMPLEXITY']

            if development:
                print()
                print(f'Starting Loop {iteration+1} for category {category}'), 

            all_patterns = set()
            for doc in docs:
                all_patterns.update(pattern_generation.extract_patterns(doc, category_lexicon, left_tokens,
                    parent_tokens, right_tokens, left_sibling, right_sibling, min_pattern_complexity, max_pattern_complexity))

            if development:
                print('1: Patterns Extracted')
                print(f'Total Patterns for Round {iteration+1}: {len(all_patterns)}')

            # Convert all pattern tuples to forms accepted by spaCy
            # then add them to the matcher
            # lastly, hash them so we can find the matches later.
            matcher = DependencyMatcher(nlp.vocab)
            hasher = {}
            new_patterns = [pattern for pattern in all_patterns if pattern not in extracted_patterns_dict]

            # If new patterns is too large then randomly sample from it instead
            if len(new_patterns) >= MAX_NEW_PATTERNS_PER_ROUND:
                new_patterns = sample(new_patterns, MAX_NEW_PATTERNS_PER_ROUND)

            # If there are no new patterns skip round 2
            if len(new_patterns) > 0:
                for pattern in new_patterns:
                    dependency_pattern = convert_to_dependency_pattern(pattern)
                    matcher.add(str(pattern), [dependency_pattern])
                    hasher[nlp.vocab.strings.add(str(pattern))] = pattern

                if development:
                    print('2: Completed Pattern Conversion')

                # Match on patterns and extract words
                progress = 0
                for doc in docs:
                    matches = matcher(doc)
                    for match in matches:
                        match_id, match_tokens = match
                        target_token = doc[match_tokens[0]]
                        target_text = target_token.text
                        pattern = hasher[match_id]
                        extracted_patterns_dict[pattern].add(target_text)
                    if development:
                        progress += 1
                        if progress % PROGRESS_SKIP == 0:
                            print(f'Made it {round((progress/len(docs))*100)}% of the way via extraction')
            else:
                if development:
                    print('2: No New Patterns on this Iteration')

            # Now extract the new patterns from the dict and add
            # the extracted text
            extracted_patterns = []
            for pattern in all_patterns:
                extracted_patterns.append((pattern, extracted_patterns_dict[pattern]))

            if development:
                print('3: Extraction Done on All Potential Patterns')

            scored_patterns = []
            for pattern_set in extracted_patterns:
                score = r_log_f(pattern_set[1], category, category_lexicon)
                scored_patterns.append((pattern_set[0], pattern_set[1], score))
            # Sort by score and then by pattern string
            scored_patterns.sort(key=lambda x: str(x[0]))
            scored_patterns.sort(key=itemgetter(2), reverse=True)
            scored_patterns = [scored_pattern for scored_pattern in scored_patterns if not scored_pattern[1].issubset(set(category_lexicon))]

            pattern_pool_size = PATTERN_POOL_INIT_SIZE + (PATTERN_POOL_SIZE_INCREASE * iteration)
            chosen_patterns = scored_patterns[:pattern_pool_size]

            # Store the best pattern that hasn't already been chosen for later analysis
            for chosen_pattern in chosen_patterns:
                # Extract the literal pattern from the pattern set
                pattern = chosen_pattern[0]
                if pattern not in top_patterns[category]:
                    top_patterns[category].add(pattern)
                    break

            candidate_words = set(itertools.chain.from_iterable([chosen_pattern[1] for chosen_pattern in chosen_patterns]))
            # Remove non alpha words
            candidate_words = {word for word in candidate_words if word.isalpha()}
            previously_selected_words = list(itertools.chain.from_iterable([lexicons[category] for category in categories]))

            if development:
                print('4: Patterns Scored and Trimmed')

            if mutual_exclusion:
                filtered_words = [word for word in candidate_words if word in previously_selected_words and word not in category_lexicon]
                if len(filtered_words) > 0:
                    print('Words Filtered By Mutual Exclusivity')
                    print(filtered_words)
                candidate_words = {word for word in candidate_words if word not in filtered_words}

            if semantic_drift:
                candidate_words = drift_filter(candidate_words, lexicons[category])

            scored_words = []
            for word in candidate_words:
                candidate_word_patterns = [pattern for pattern in extracted_patterns if word in pattern[1]]
                score = avg_log(candidate_word_patterns, category, category_lexicon)
                scored_words.append((word, score))
            # Sort by score and then by word
            scored_words.sort(key=itemgetter(0))
            scored_words.sort(key=itemgetter(1), reverse=True)
            scored_words = [word for word in scored_words if word[0].isalpha()]
            chosen_words = [word[0].lower() for word in scored_words[:WORDS_PER_ROUND]]
            category_lexicon += chosen_words

            if development:
                print('5: Words Scored and Trimmed')
                print(f'This Rounds Words: {chosen_words}')
                print()

            # Save pattern extractions for next time
            # Make sure it saves after every iteration
            with open(pickle_path, 'wb') as file: 
                pickle.dump(extracted_patterns_dict, file) 
    
    for category in categories:
        category_seeds = seeds[category]
        category_lexicon = lexicons[category]
        extracted_words = [word for word in category_lexicon if word not in category_seeds]
        patterns = top_patterns[category]

        # Write the output of generated words
        file = open(f'{output}{category}.txt','w')
        for word in extracted_words:
            file.write(f'{word}\n')
        file.close()

        # Write top patterns to output
        file = open(f'{output}{category}-patterns.p','wb')
        pickle.dump(patterns, file)
        file.close()

    print("End Time:", datetime.now().strftime("%H:%M:%S"))