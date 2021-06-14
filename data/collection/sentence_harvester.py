from googlesearch import search
import googlesearch
import requests
import bs4
from bs4 import BeautifulSoup
import json
import time
import re
import urllib.error
import itertools
import random

split = str.split

def filter_sent(sent, c):
    if c in sent and 50 > len(split(sent)) > 7:
        return sent

def faster_harvest_sentences(compounds, limit, unsuccessful_compounds=None, compound_sentences=None):
    if compound_sentences == None:
        compound_sentences = {}

    if unsuccessful_compounds == None:
        unsuccessful_compounds = []

    print(f'Compound sentences already harvested: {len(compound_sentences)}')
    print(f'Compounds that we couldn\'t find sentences for so far: {len(unsuccessful_compounds)}')
    compounds = [c for c in compounds if c not in compound_sentences and c not in unsuccessful_compounds]
    print(f'Compounds to go: {len(compounds)}')
    random.shuffle(compounds)

    lower = str.lower
    find_all = BeautifulSoup.find_all
    strip = str.strip
    get = requests.get
    getText = bs4.element.Tag.getText
    sub = re.sub
    split = re.split

    whitespace_reg = re.compile('\s+')
    split_reg = re.compile('\. |\?+|!+')

    it = iter(compounds[:limit])
    iterations = 0
    for c in it:
        iterations += 1
        start_time = time.time()
        try:
            response = search('"' + c + '"', start=80, stop=150, user_agent=googlesearch.get_random_user_agent(), pause=5.0)
        except:
            print(f'Unsuccessful search for "{c}"')
            unsuccessful_compounds.append(c)
            continue
        for url in itertools.islice(response, 100):
            user_agent = googlesearch.get_random_user_agent()
            headers = {'User-Agent': user_agent}
            try:
                page = get(url, timeout=5, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                lines = [sub(whitespace_reg, ' ', lower(getText(t))) for t in find_all(soup, 'p')]
            except:
                continue
            if not lines:
                continue
            sentences = [sent for line in lines for sent in split(split_reg, strip(line))]
            filtered_sents = list(filter(lambda s: filter_sent(s, lower(c)), sentences))

            if filtered_sents:
                print(f'Found sentence(s) for "{c}"')
                try:
                    compound_sentences[c] = compound_sentences[c] + filtered_sents
                except:
                    compound_sentences[c] = filtered_sents
            if c in compound_sentences and len(compound_sentences[c]) >= 3:
                break
        if c not in compound_sentences:
            print(f'Could not find a sentence for "{c}"')
            unsuccessful_compounds.append(c)
        end_time = time.time()
        print(f'Iteration number {iterations} took {end_time - start_time} seconds!')
        if iterations % 2 == 0:
            with open('compound_sents.json', 'w') as outfile:
                json.dump(compound_sentences, outfile)

    return compound_sentences, unsuccessful_compounds

def main():
    with open('all_compounds.txt', 'r') as compound_file:
        compounds = [l.rstrip(' \n') for l in compound_file.readlines()]

    try:
        with open('compound_sents.json', 'r') as infile:
            compound_sentences = json.load(infile, encoding='utf-8')
    except:
        compound_sentences = None

    try:
        with open('unsuccessful_compounds.txt', 'r') as uns_in:
            unsuccessful_compounds = [l.rstrip(' \n') for l in uns_in.readlines()]
    except:
        unsuccessful_compounds = None

    i = 0
    while i < 20:
        try:
            new_compound_sentences, unsuccessful_compounds = faster_harvest_sentences(
                compounds,
                10,
                unsuccessful_compounds=unsuccessful_compounds,
                compound_sentences=compound_sentences
            )
            with open('compound_sents.json', 'w', encoding='utf-8') as outfile:
                json.dump(new_compound_sentences, outfile)
            with open('unsuccessful_compounds.txt', 'w') as uns_out:
                for c in unsuccessful_compounds:
                    uns_out.write(c + ' \n')
            i += 1
            print(f'########## PROCESSED {i*10} COMPOUNDS ##########')
        except urllib.error.HTTPError:
            sleep_time = 240
            print(f'Too many requests. Going to sleep for {sleep_time} seconds.')
            time.sleep(sleep_time)
        except:
            continue

if __name__ == '__main__':
    main()