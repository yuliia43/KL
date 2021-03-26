import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_defaults():
    return pd.read_csv('stats.csv', sep=';', encoding='windows-1251', index_col=0)


def read_file(filename):
    text = ''
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            text += line.upper()
    return text


def count_grams(text):
    ngrams = {}

    words_tokens = nltk.word_tokenize(text)
    for word in words_tokens:
        for i in range(len(word) - 1):
            seq = word[i]
            if seq not in ngrams.keys():
                ngrams[seq] = 0
            ngrams[seq] += 1
    return ngrams


def count_bigrams(text):
    ukr_letters_str = "АБВГҐДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩІЇЬЄЮЯ"
    ukr_letters = [char for char in ukr_letters_str]
    ngrams = pd.DataFrame(index=ukr_letters, columns=ukr_letters).fillna(0)

    words_tokens = nltk.word_tokenize(text)
    for word in words_tokens:
        for i in range(len(word) - 2):
            if word[i] in ukr_letters and word[i+1] in ukr_letters:
                ngrams.loc[word[i], word[i+1]] += 1
    ngrams = ngrams/ngrams.to_numpy().sum()
    return ngrams


def get_bigram_stats():
    stats = pd.read_csv('bigrams_stats.csv', sep=';', encoding='windows-1251', index_col=0)
    stats = stats / stats.to_numpy().sum()
    return stats


def bigrams_difference(bigram, stats):
    delta = bigram - stats
    return delta.abs()


def append_stats(defs, author, gram):
    table = defs.copy()
    for char, value in gram.items():
        if char in table.index:
            table.loc[char, author] = value
    return table


def plot(bigram, author):
    Z = bigram.to_numpy()
    x = bigram.columns
    y = bigram.index

    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z)
    ax.set_title(author)
    plt.show()


def test_gram(grams_df, bigrams):
    # copy = grams_df[grams_df.columns.intersection(['Deresh','Karpa','Andruhovich'])]
    sizes = [5000, 10000, 25000, 50000]
    test_text = read_file("Texts/Deresh_test.txt")
    for size in sizes:
        fragment = test_text[:size]
        # grams_analysis(copy, fragment)
        bigrams_analysis(bigrams, fragment)


def bigrams_analysis(bigrams, fragment):
    bigram_test = count_bigrams(fragment)
    min = 100000
    author_name = ""
    sums = []
    for name, bigram in bigrams.items():
        diff = bigrams_difference(bigram, bigram_test)
        sum = diff.to_numpy().sum()
        sums.append(sum)
        if sum < min:
            min = sum
            author_name = name
    print(sums, author_name)


def grams_analysis(copy, fragment):
    dict = count_grams(fragment)
    if 'Ґ' in dict.keys():
        dict['Г'] += dict['Ґ']
    gram = pd.DataFrame(dict.values(), index=dict.keys())
    gram = gram.loc[grams_df.index]
    sum = gram.sum()
    gram = gram / sum
    diff = copy.to_numpy() - gram.to_numpy()
    abs_sum = np.abs(diff).sum(axis=0)
    idx = np.argmin(abs_sum)
    res = copy.columns[idx]
    print(abs_sum, res)


def gram_analysis():
    global i, file, text, grams_df, author_name
    for i, file in enumerate(files):
        grams_df[file] = ""
        text = read_file("Texts/" + file + ".txt")
        gram = count_grams(text)
        if 'Ґ' in gram.keys():
            gram['Г'] += gram['Ґ']
        grams_df = append_stats(grams_df, file, gram)
        if (i % 3 == 2):
            author_name = file.split()[0]
            grams_df[author_name] = grams_df.loc[:, [c for c in grams_df.columns if c.split()[0] == author_name]].sum(
                axis=1)
    sums = grams_df.sum()
    sums.to_csv('sums.csv', sep=';')
    grams_df = grams_df / sums
    grams_df.to_csv('df.csv', sep=';',float_format='%.7f')


if __name__ == '__main__':

    stats = get_bigram_stats()

    grams_df = read_defaults()
    files = ["Deresh - Golova Iakova",
             "Deresh - Kult",
             "Deresh - Namir",
             "Karpa - Bіtches Get Everythіng",
             "Karpa - Froid by plakav",
             "Karpa - 50 khvylyn travy",
             "Andruhovich - Lito Mileni",
             "Andruhovich - Stari ludi",
             "Andruhovich - Feliks Avstrya",
             ]
    gram_analysis()

    bigrams = {}
    text = ""
    for i, file in enumerate(files):
        text += read_file("Texts/"+file+".txt")
        if i % 3 == 2:
            bigram = count_bigrams(text)
            author_name = file.split()[0]
            plot(bigram, author_name)
            diff = bigrams_difference(bigram, stats)
            plot(diff, author_name + ' difference')
            bigrams[author_name] = bigram
            text = ""
    test_gram(grams_df, bigrams)
