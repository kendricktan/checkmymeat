from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
from checker.forms import ImageForm
import checker.forest as forest

from skimage import io

import cv2
import numpy as np

# Good RGB for 

def ColorDistance(rgb1,rgb2):
    '''d = {} distance between two colors(3)'''
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5
    return d

# Create your views here.
@csrf_exempt
def index_view(request):
    if request.method == 'POST':        
        url = request.POST['image_url']                
        ctx = forest.predict(url)      
        
        # Raw values
        return JsonResponse(ctx)            

    form = ImageForm()
    return render(request, 'index.html', {'form': form})
