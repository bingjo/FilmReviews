import re
import fasttext


model = fasttext.load_model('model.bin')
reg = re.compile('[^a-zA-Z ]')


class GetRating:

    d_rating = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}

    def __init__(self, text):
        self.text = text

    @staticmethod
    def preprocessing(text: str):
        # Удаление всех символов, кроме английских букв
        text = reg.sub('', text)
        # Приведение текста к нижнему регистру
        text = text.lower()
        # Замена тега перноса строки
        text = text.replace('<br />', ' ')
        # Удаление лишних пробеллов
        text = re.sub(' +', ' ', text)
        return text

    # Предсказание оценки
    @staticmethod
    def model_predict(text: str):
        r = model.predict(text)
        r = GetRating.d_rating[int(str(r[0][0]).split('__label__')[1])]
        if r <= 4:
            r_str = False
        if r >= 7:
            r_str = True
        return r, r_str

    # Параметры для отображения на странице
    def get_rating(self):
        sentiment = False
        error = True
        text_rating = 'Рекомендуемая оценка: '
        if len(self.text) == 0:
            text_rating = 'Введите текст!'
        elif 0 < len(self.text) < 30:
            text_rating = 'Введите текст большей длины!'
        else:
            rating, sentiment = GetRating.model_predict(GetRating.preprocessing(self.text))
            text_rating += str(rating)
            error = False
        GetRating.model_predict(self.text)
        return text_rating, sentiment, error
