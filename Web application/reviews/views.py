from django.shortcuts import render
from .service.get_rating import GetRating


# Create your views here.
def index(request):
    data = {'text_rating': '', 'text': '', 'sentiment': False, 'error': True}
    if 'button' in request.POST:
        text = request.POST.get('comment')
        data['text'] = text
        try:
            rating, sentiment, error = GetRating(text).get_rating()
            data['text_rating'] = rating
            data['sentiment'] = sentiment
            data['error'] = error
        except:
            data['text_rating'] = 'В процессе оценки произошла ошибка :('
        return render(request, 'reviews/index.html', context=data)
    return render(request, 'reviews/index.html', context=data)
