from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from checker.forms import ImageForm

# Create your views here.
def index_view(request):
    if request.method == 'POST':        
        print(request.POST['image_url'])
        context = {'type': 'blue rare'}
        return JsonResponse(context)            

    form = ImageForm()
    return render(request, 'index.html', {'form': form})
