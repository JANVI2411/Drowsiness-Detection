from django.urls import path
from . import views

urlpatterns=[
    path('',views.home, name = "home"),
    # path('open_webcam/', views.WebCam, name = "webcam"),
    path('webcam/', views.webcam_second, name = "webcam")
]
