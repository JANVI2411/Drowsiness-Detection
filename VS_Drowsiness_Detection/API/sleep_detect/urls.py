from django.urls import path
from . import views

urlpatterns=[
    path('',views.home, name = "home"),
    path('open_webcam/', views.WebCam, name = "open_webcam"),
    path('demo_webcam/',views.demo_webcam,name='demo_webcam'),
]
