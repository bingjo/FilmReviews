import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from models import TextRNN, TrainModels, BertModel, TrainBertModel, TextCNN
from tensorflow.keras.utils import to_categorical


# Препроцессинг текстов
class PreprocessingData:

    reg = re.compile('[^a-zA-Z ]')
    ggg = 0

    def __init__(self, train_list: list, test_list: list):
        self.texts, self.labels = PreprocessingData.get_texts_labels(train_list, test_list)
        self.max_features = 5000
        self.max_len = 152
        self.embedding_size = 128

    def bin_data_preprocessing(self):
        pass

    def cat_data_preprocessing(self):
        pass

    # Получение текстов и соответствующих меток
    @staticmethod
    def get_texts_labels(train1: list, test1: list):
        texts = []
        labels = []
        for i in range(len(train1)):
            t = train1[i].split('\t')
            texts.append(PreprocessingData.basic_preprocessing(t[1]))
            labels.append(int(t[0]))
        for i in range(len(test1)):
            t = test1[i].split('\t')
            texts.append(PreprocessingData.basic_preprocessing(t[1]))
            labels.append(int(t[0]))
        return texts, labels

    # Базовая предварительная обработка,
    # которая используется во всех моделях
    @staticmethod
    def basic_preprocessing(text: str):
        # Удаление всех символов, кроме английских букв
        text = PreprocessingData.reg.sub('', text)
        # Приведение текста к нижнему регистру
        text = text.lower()
        # Замена тега перноса строки
        text = text.replace('<br />', ' ')
        # Удаление лишних пробеллов
        text = re.sub(' +', ' ', text)
        # Удаление стоп слов
        # words = text.split(' ')
        # filtered_words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
        return text


# Предварительная обработка для моделей RNN и CNN
class TheFirstPreprocessing(PreprocessingData):
    def tokenizer(self):
        # Приведение текстов и меток в масссивы чисел
        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(self.texts)
        texts_seq = tokenizer.texts_to_sequences(self.texts)
        texts_mat = sequence.pad_sequences(texts_seq, maxlen=self.max_len)
        x_train, x_test, y_train, y_test = train_test_split(texts_mat, self.labels, random_state=42, train_size=0.7)
        return x_train, x_test, y_train, y_test

    # Пред. обработка для бинарной классификации
    def bin_data_preprocessing(self):
        for i in range(len(self.labels)):
            if self.labels[i] <= 3:
                self.labels[i] = 0
            else:
                self.labels[i] = 1
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        x_train, x_test, y_train, y_test = TheFirstPreprocessing.tokenizer(self)
        return x_train, x_test, y_train, y_test

    # Пред. обработка для многоклассовой классификации
    def cat_data_preprocessing(self):
        x_train, x_test, y_train, y_test = TheFirstPreprocessing.tokenizer(self)
        y_train = to_categorical(y_train, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)
        return x_train, x_test, np.asarray(y_train), np.asarray(y_test)


# Предварительная обработка для обучения различных моделей BERT
# с использованием бибилотеки Simple Transformers
class BertPreprocessing(PreprocessingData):
    # Приведение текстов и меток в DataFrame формат
    def tokenizer(self):
        x_train, x_test, y_train, y_test = train_test_split(self.texts, self.labels, random_state=42, train_size=0.7)
        train_data = []
        test_data = []
        for i in range(len(x_train)):
            train_data.append([x_train[i], y_train[i]])
        for i in range(len(x_test)):
            test_data.append([x_test[i], y_test[i]])
        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text", "labels"]
        test_df = pd.DataFrame(test_data)
        test_df.columns = ["text", "labels"]
        return train_df, test_df

    # Пред. обработка для бинарной классификации
    def bin_data_preprocessing(self):
        for i in range(len(self.labels)):
            if self.labels[i] <= 3:
                self.labels[i] = 0
            else:
                self.labels[i] = 1
        train_df, test_df = BertPreprocessing.tokenizer(self)
        return train_df, test_df

    # Пред. обработка для многоклассовой классификации
    def cat_data_preprocessing(self):
        train_df, test_df = BertPreprocessing.tokenizer(self)
        return train_df, test_df


# Предварительная обработка текстов для fastText
class FastTextPreProcessing(PreprocessingData):
    # Запись текстовых файлов с метками и текстами
    def tokenizer(self):
        train_df = open('data/fasttext/train.txt', 'w', encoding='utf8')
        test_df = open('data/fasttext/test.txt', 'w', encoding='utf8')
        test_labels = open('data/fasttext/test_labels.txt', 'w', encoding='utf8')
        x_train, x_test, y_train, y_test = train_test_split(self.texts, self.labels, random_state=42, train_size=0.7)
        for i in range(len(x_train)):
            t = '__label__' + str(y_train[i]) + ' ' + x_train[i]
            if i == len(x_train) - 1:
                train_df.write(t)
            else:
                train_df.write(t + '\n')
        for i in range(len(x_test)):
            t = x_test[i]
            ll = str(y_test[i])
            if i == len(x_test) - 1:
                test_df.write(t)
                test_labels.write(ll)
            else:
                test_df.write(t + '\n')
                test_labels.write(ll + '\n')
        train_df.close()
        test_df.close()

    # Пред. обработка для бинарной классификации
    def bin_data_preprocessing(self):
        for i in range(len(self.labels)):
            if self.labels[i] <= 3:
                self.labels[i] = 0
            else:
                self.labels[i] = 1
        FastTextPreProcessing.tokenizer(self)

    # Пред. обработка для многоклассовой классификации
    def cat_data_preprocessing(self):
        FastTextPreProcessing.tokenizer(self)


if __name__ == '__main__':
    user_model = 'FastText'
    # Открытие набора данны
    file_train = open('data/train.txt', 'r', encoding='utf8').read().split('\n')
    file_test = open('data/test.txt', 'r', encoding='utf8').read().split('\n')

    # Получение выборок, моделей НС и их обучение
    if user_model == 'BERT':
        train, test = BertPreprocessing(file_train, file_test).cat_data_preprocessing()
        model = BertModel(architecture='bert', model_id='bert-base-uncased', outputs=8).get_model()
        TrainBertModel(x_train=train, x_test=test, model=model).train_model()

    if user_model == 'Basic models':
        X_train, X_test, Y_train, Y_test = TheFirstPreprocessing(file_train, file_test).cat_data_preprocessing()
        model = TextCNN(max_len=152, max_features=5000, embedding_size=128, pool_size=4,
                        filters=32, outputs=8).get_model()
        TrainModels(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test, batch_size=32,
                    epochs=5, losses='categorical_crossentropy', model=model).train_model()

    if user_model == 'FastText':
        FastTextPreProcessing(file_train, file_test).cat_data_preprocessing()
