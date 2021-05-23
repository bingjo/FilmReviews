# Оценка рецензий на фильмы

## Оглавление

1.	[Набор данных](#набор-данных)
2.	[Обучение нейронных сетей](#обучение-нейронных-сетей)
3.	[Веб-сервис](#веб-сервис)

## Набор данных

Для обучение НС был использован набор данных [aclImdb](https://github.com/gizenmtl/IMDB-Sentiment-Analysis-and-Text-Classification/tree/master/aclImdb).

Набор данных содержит 50 тыс. размеченных отзывов на фильмы. Отзывы на английском языке.

Также в наборе данных отсутсвуют отзывы с оценкой 5 и 6, так как возникает проблема отнесения данных отзывов в одну из двух категорий.

## Обучение нейронных сетей

Было выбрано пять моделей НС:

1.	[Bert Base](https://arxiv.org/abs/1810.04805v2);
2.	[Bert Large](https://arxiv.org/abs/1810.04805v2);
3.	[fastText](https://arxiv.org/abs/1607.01759);
4.	[TextCNN](https://www.aclweb.org/anthology/D14-1181/);
5.	[TextRNN](https://www.ijcai.org/Proceedings/16/Papers/408.pdf).

Результаты обучения моделей НС:

|Модель|Бинарная классификация|Многоклассовая классификация|
|:----:|:---------------------:|:-------------------------:|
|Bert Base|n/d|41.16%|
|Bert Large|n/d|42.74%|
|fastText|n/d|**45.6%**|
|TextCNN|n/d|41.63%|
|TextRNN|n/d|41.95%|

В разработанном веб-сервисе использвалась модель fastText, так как имеет самую большую точность (45.6%).

[Исходный код](https://github.com/bingjo/FilmReviews/tree/main/Neural%20network%20models).

Для обучения и тестирования моделей НС использовались [библиотеки python](https://github.com/bingjo/FilmReviews/blob/main/Neural%20network%20models/requirements.txt).

## Веб-сервис

Сервис создан на базу фреймворка Django.
