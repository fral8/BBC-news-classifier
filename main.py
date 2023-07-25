import io
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd

NUM_WORDS = 10000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8


def remove_stopwords(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (string): sentence to remove the stopwords from

    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)

    return sentence

def load_data(path):
    df=pd.read_csv(path)
    dataset=df['text']
    labels=df['category']
    train_set=dataset[0:int(len(dataset)*TRAINING_SPLIT)]
    train_label=labels[0:int(len(labels)*TRAINING_SPLIT)]
    test_set=dataset[int(len(dataset)*TRAINING_SPLIT):]
    test_label=labels[int(len(labels)*TRAINING_SPLIT):]
    return train_set,train_label,test_set,test_label

def preprocessing(train_set,test_set):
    tokenizer=Tokenizer(NUM_WORDS,OOV_TOKEN)
    tokenizer.fit_on_texts(train_set)
    train_sequences=tokenizer.texts_to_sequences(train_set)
    test_sequences = tokenizer.texts_to_sequences(test_set)

    train_padded=pad_sequences(train_sequences,maxlen=MAXLEN,truncating=PADDING)
    test_padded=pad_sequences(test_sequences,maxlen=MAXLEN,truncating=PADDING)
    return train_padded,test_padded

def create_model():
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAXLEN),
        tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def label_preprocessing(training_labels,test_labels):
    label_tokenizer = Tokenizer()

    # Fit the tokenizer on all the labels
    label_tokenizer.fit_on_texts(training_labels)

    # Convert labels to sequences
    label_seq = label_tokenizer.texts_to_sequences(training_labels)

    # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
    label_seq_np_tr = np.array(label_seq) - 1
    label_seq_test = label_tokenizer.texts_to_sequences(test_labels)

    # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
    label_seq_np_test = np.array(label_seq_test) - 1

    return label_seq_np_tr,label_seq_np_test

train_set,train_label,test_set,test_label=load_data("C:\Progetti\Personali\MachineLearning\\nlp\Coursera\BBC_news\Data\\bbc-text.csv")
for sentence in train_set:
    sentence=remove_stopwords(sentence)

for sentence in test_set:
    sentence=remove_stopwords(sentence)

train_set,test_set=preprocessing(train_set,test_set)
label_seq_np_tr,label_seq_np_test=label_preprocessing(train_label,test_label)
model=create_model()
history=model.fit(train_set,label_seq_np_tr,validation_data=(test_set,label_seq_np_test),epochs=30)


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")