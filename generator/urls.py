from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='gen-home'),
    path('about/', views.about, name='gen-about'),
    path('images/',views.images, name='gen-images')
]