from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("suggest/", views.home, name="suggest"),  # Reuse the home view for handling suggestions
]
