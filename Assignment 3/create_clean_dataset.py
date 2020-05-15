# http://scikit-learn.org/stable/datasets/twenty_newsgroups.html

def create_clean_dataset(max_features):
    """
    Create clean data set of 20 news data set, it uses stemming, eliminating stop words and ...
    :param max_features:
    :return:
    """
    import glob
    import numpy as np
    import nltk as nlp
    import os
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer
    import json

    #nlp.download()

    # list of news categories
    categories = []
    # dictionary of news that belongs to different categories
    dic_news_cat = {}

    # all word of dataset
    #dataset_word = []

    # number of documents
    num_doc = 0
    num_doc_cat = {}

    # name of different categories
    for news_path in glob.glob(os.path.join('dataset', '20news-18828', '*')):
        # add category to list of categories
        categories.append(news_path.split('\\')[2])
        dic_news_cat[news_path.split('\\')[2]] = []
        num_doc_cat[news_path.split('\\')[2]] = 0

    punc_table = str.maketrans('', '', string.punctuation)
    stop_words = set(nlp.corpus.stopwords.words('english'))
    words_en = set(nlp.corpus.words.words('en'))

    # read new of each category
    for cat_index, news_category in enumerate(categories):
        for news_path in glob.glob(os.path.join('dataset', '20news-18828', news_category, '*')):

            # count number of documents
            num_doc += 1
            num_doc_cat[news_category] += 1

            # open file and read text
            file = open(news_path, 'rt')
            text = file.read()
            file.close()

            # split into words
            words = nlp.tokenize.word_tokenize(text)

            # normalize by converting to lower case
            words = [w.lower() for w in words]

            # remove punctuation from each word
            stripped = [w.translate(punc_table) for w in words]

            # remove words that are not alphabetic
            words = [word for word in stripped if word.isalpha()]

            # filter out stop words
            words = [w for w in words if w not in stop_words]

            # stemming of words
            porter = nlp.stem.porter.PorterStemmer()
            clean_text_arr = [porter.stem(word) for word in words]

            # clean non english words
            #clean_text_arr = [w for w in clean_text_arr if w not in words_en]

            #
            #print(clean_text_arr[:100])

            # save clean text beside other text with same category
            dic_news_cat[news_category].append(' '.join(clean_text_arr))

            # add new words to collection of all words in entire categories
            #dataset_word = np.unique(dataset_word+clean_text_arr).tolist()

        print('Category {}/{}'.format(cat_index, len(categories)))

    #print(len(dataset_word))

    # initial bag of words
    #bag_of_words = np.zeros(shape=(num_doc, len(dataset_word)))

    # each element is a string(text of a file)
    X = []
    # each element is a number which determines true class of text
    y = []
    # determines each number equals which class
    label_class = {}
    for cat_index, news_category in enumerate(categories):

        label_class[cat_index] = news_category

        for text_file in dic_news_cat[news_category]:
            X.append(text_file)
            y.append(cat_index)

    #print(X)
    #print(y)
    #print(label_class)

    # create bag of words and create features
    tfid = TfidfVectorizer(max_features=max_features, max_df=0.30, min_df=10)
    X_mapped = tfid.fit_transform(raw_documents=X)

    X_mapped = np.array(X_mapped.toarray())
    #print(X_mapped)
    y = np.array(y).reshape(len(y), 1)
    #print(y)
    #print('Number of documents in each category:')
    #print(num_doc_cat)

    data = (X_mapped, y, label_class)

    return data