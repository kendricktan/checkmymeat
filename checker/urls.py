from django.conf.urls import url
from django.contrib import admin

from checker.views import index_view

urlpatterns = [
    url(r'^$', index_view, name='index_view'),    
]