from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='gen-home'),
    path('about/', views.about, name='gen-about'),
    path('images/',views.images, name='gen-images'),
    path('aboutproject/', views.aboutproject, name='gen-aboutproject'),
    path('future/', views.future, name='gen-future'),
    path('aboutdevelopers/',views.aboutdevelopers, name='gen-aboutdevelopers')
]