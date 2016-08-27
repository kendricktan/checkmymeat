from django import forms

class ImageForm(forms.Form):
    image_url = forms.CharField(max_length=None)