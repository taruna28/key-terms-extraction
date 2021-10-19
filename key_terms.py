import string

import nltk
import pandas as pd
from lxml import etree
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemma = WordNetLemmatizer()
punctuation = list(string.punctuation)
stopwords_ = stopwords.words("english")

root = etree.parse("news.xml").getroot()[0]
stories = list()

for i in root:  # First, we must reduce, remove punctuation / stopwords, and only pick the nouns before TF-IDF
    story = i[1].text
    word_tokenizer = nltk.word_tokenize(story.lower())
    word_tokenizer = [lemma.lemmatize(word) for word in word_tokenizer]
    word_tokenizer = [word for word in word_tokenizer if word not in punctuation]
    word_tokenizer = [word for word in word_tokenizer if word not in stopwords_]
    word_tokenizer = [word for word in word_tokenizer if nltk.pos_tag([word])[0][1] == "NN"]
    stories.append(" ".join(word_tokenizer))

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(stories).toarray()
# print(matrix)
terms = vectorizer.get_feature_names()  # The vocab of all the stories

current_story = 0  # This will be used to loop through the score matrix / title of each news story
for i in root:
    df = pd.DataFrame()
    df["Words"] = terms
    df["Score"] = matrix[current_story]  # Score is based off of our array of TF-IDF scores
    df.sort_values(["Score", "Words"], ascending=False, inplace=True)  # Sorts the words by TF-IDF score
    data = df.iloc[0:5]["Words"]  # Gets the 5 most common words into a series (list)
    print("{}:".format(i[0].text))  # title
    for word in data:
        print(word, end=" ")  # 5 most common words in dataframe's word series
    print("\n")
    current_story += 1
