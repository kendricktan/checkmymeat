from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from checker.forms import ImageForm

from skimage import io
from checker.predict import predict, WIDTH, HEIGHT, DEPTH, json_settings

import cv2
import numpy as np

# Create your views here.
def index_view(request):
    if request.method == 'POST':        
        url = request.POST['image_url']                
        image = io.imread(url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (1, HEIGHT, WIDTH, DEPTH))

        # Raw values
        results_raw = predict(image)
        print(results_raw)
        results_index = np.argmax(results_raw)
        results_score = results_raw[0][results_index]*100
        results_str = json_settings['level'][str(results_index)]
        context = {'type': results_str, 'results_score': results_score}
        return JsonResponse(context)            

    form = ImageForm()
    return render(request, 'index.html', {'form': form})
