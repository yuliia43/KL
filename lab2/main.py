import nltk
import pandas as pd
import math
import matplotlib.pyplot as plt


def read_from_file(filename):
    df = pd.read_csv(filename, sep=';', encoding='windows-1251', index_col=0)
    return df


def preprocess_text(row):
    txt = row[['Анекдот']].values[0]
    txt = txt.lower()
    txt = txt.replace(',', '')
    txt = txt.replace('.', '')
    txt = txt.replace(';', '')
    txt = txt.replace('- ', '')
    txt = txt.replace('-', '')
    txt = txt.replace('–', '')
    txt = txt.replace('—', '')
    txt = txt.replace('"', '')
    txt = txt.replace("'", '')
    txt = txt.replace('\r', ' ')
    txt = txt.replace('\n', ' ')
    txt = txt.replace('  ', ' ')
    return txt


def teach_nbk(file_name, min_size=0):
    df = read_from_file(file_name)
    words_ny = []
    words_love = []
    for index, row in df.iterrows():
        txt = preprocess_text(row)
        if row[['Клас']].values[0] == 0:
            words_ny.extend(txt.split(' '))
        else:
            words_love.extend(txt.split(' '))
    words_ny = [word for word in words_ny if len(word) > min_size]
    words_love = [word for word in words_love if len(word) > min_size]
    fdist_ny = nltk.FreqDist(words_ny)
    fdist_love = nltk.FreqDist(words_love)
    fdist = nltk.FreqDist(words_love+words_ny)
    return fdist_ny, fdist_love, fdist.N()


def read_test_data():
    df = read_from_file("jokes.csv")
    texts_ny = []
    texts_love = []
    for index, row in df[60:].iterrows():
        txt = preprocess_text(row)
        if row[['Клас']].values[0] == 0:
            texts_ny.append(txt.split(' '))
        else:
            texts_love.append(txt.split(' '))
    return texts_ny, texts_love


def classificate(fdist_ny, fdist_love, k, test, test_class, debug=0):
    classificated_right = 0
    k_ny, sum_words_ny = count_vars(fdist_ny)
    k_love, sum_words_love = count_vars(fdist_love)
    if debug == 1:
        print('Клас "новий рік". Кількість унікальних слів: %s.'
              ' Сумарна кількість слів в документах: %s' %(k_ny, sum_words_ny))
        print('Клас "кохання". Кількість унікальних слів: %s.'
              ' Сумарна кількість слів в документах: %s' %(k_love, sum_words_love))
    for data in test:
        p_ny = math.log(0.5)
        p_love = math.log(0.5)
        for word in data:
            p_ny += count_word_prob_for_class(fdist_ny, k, sum_words_ny, word)
            p_love += count_word_prob_for_class(fdist_love, k, sum_words_love, word)
            if debug == 1:
                print('Слово: %s. Вірогідність класу "новий рік": %s.'
                      '  Вірогідність класу "кохання": %s' % (word, p_ny, p_love))
        if p_ny >= p_love:
            defined_class = 0
        else:
            defined_class = 1
        if test_class == defined_class:
            classificated_right+=1
    return classificated_right


def count_word_prob_for_class(fdist, k, sum_words, word):
    wic = 0
    if word in fdist.keys():
        wic = fdist[word]
    p_word = math.log((wic + 1) / (k + sum_words))
    return p_word


def count_vars(fdist):
    sum_words = fdist.N()
    k = len(fdist)
    return k, sum_words

def most_common_words():
    fdist_ny, fdist_love, _ = teach_nbk("jokes.csv", min_size=0)
    ny_common1 = fdist_ny.most_common(10)
    love_common1 = fdist_love.most_common(10)
    fdist_ny, fdist_love, _ = teach_nbk("jokes.csv", min_size=3)
    ny_common2 = fdist_ny.most_common(10)
    love_common2 = fdist_love.most_common(10)
    draw_plot(ny_common1, "10 найбільш часто використовуваних слів в в категорії 'новий рік'")
    draw_plot(love_common1, "10 найбільш часто використовуваних слів в в категорії 'кохання'")
    draw_plot(ny_common2, "10 найбільш часто використовуваних слів в в категорії 'новий рік' без стоп-слів")
    draw_plot(love_common2, "10 найбільш часто використовуваних слів в в категорії 'кохання' без стоп-слів")



def draw_plot(list_of_tuples, label):
    words = [tuple[0] for tuple in list_of_tuples]
    counter = [tuple[1] for tuple in list_of_tuples]
    plt.title(label)
    plt.bar(words, counter)
    plt.show()


def main():
    plt.rcParams.update({'font.size': 8})
    # most_common_words()
    test_ny, test_love = read_test_data()
    for i in range(10, 31, 10):
        fdist_ny, fdist_love, k = teach_nbk('test_'+str(i)+'.csv')
        right = 0
        right1 = classificate(fdist_ny, fdist_love, k, test_ny, 0)
        right2 = classificate(fdist_ny, fdist_love, k, test_love, 1)
        print('Розмір навчальної вибірки: %s. Відгадано правильно: %s новорічних і %s анекдотів про кохання' % (i, right1, right2))
    # fdist_ny_10, fdist_love_10, k_10 = teach_nbk('test_10.csv')
    # fdist_ny_20, fdist_love_20, k_20 = teach_nbk('test_20.csv')
    # fdist_ny_30, fdist_love_30, k_30 = teach_nbk('test_30.csv')
    # right = classificate(fdist_ny_30, fdist_love_30, k_30, test_ny, 0)
    # print('Відгадано правильно: %s анекдотів' %right)
    # right = classificate(fdist_ny_30, fdist_love_30, k_30, test_love, 1)
    # print('Відгадано правильно: %s анекдотів' %right)


if __name__ == '__main__':
    main()
