from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, Concatenate, Dropout, MaxPooling1D, LSTM, GlobalMaxPooling1D
from simpletransformers.classification import ClassificationModel


class Models:
    def __init__(self, max_len=None, max_features=None, embedding_size=None, pool_size=None,
                 filters=None, outputs=None, architecture=None, model_id=None):
        # Для классическиз моделей:
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_size = embedding_size
        self.pool_size = pool_size
        self.filters = filters
        self.outputs = outputs
        self.architecture = architecture
        self.model_id = model_id

    def get_model(self):
        pass


class TextRNN(Models):
    def get_model(self):
        input = Input((self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_size, input_length=self.max_len)(input)
        rnn = LSTM(128)(embedding)
        output = Dense(self.outputs, activation='sigmoid')(rnn)
        model = Model(inputs=input, outputs=output)
        return model


class TextCNN(Models):
    def get_model(self):
        input = Input((self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_size, input_length=self.max_len)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        x = Dropout(0.5)(x)
        output = Dense(self.outputs, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=output)
        return model


class BertModel(Models):
    def get_model(self):
        model = ClassificationModel(self.architecture, self.model_id, num_labels=self.outputs, use_cuda=False,
                                    args={'fp16': False, 'num_train_epochs': 3, 'no_cache': True, 'no_save': True,
                                          'overwrite_output_dir': True})
        return model


class TrainModels:
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None,
                 batch_size=None, epochs=None, losses=None, model=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.losses = losses

    def train_model(self):
        self.model.compile('adam', self.losses, metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)
        scores = self.model.evaluate(self.x_test, self.y_test, batch_size=32)
        print(scores)
        print('Точность: ' + str(round(scores[1] * 100, 2)) + '%')


class TrainBertModel(TrainModels):
    def train_model(self):
        self.model.train_model(self.x_train)
        result, model_outputs, wrong_predictions = self.model.eval_model(self.x_test)
        try:
            acc = (result['tp'] + result['tn']) / (result['tp'] + result['tn'] + result['fp'] + result['fn'])
            print(result)
            print('Точность: ' + str(round(acc * 100, 2)) + '%')
        except:
            acc = result['mcc']
            print(result)
            print('Точность: ' + str(round(acc * 100, 2)) + '%')
