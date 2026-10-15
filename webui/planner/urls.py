from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("runs/new/", views.new_run, name="new_run"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("runs/<int:pk>/dispatch/", views.run_dispatch, name="run_dispatch"),
    path("compare/", views.compare, name="compare"),
    path("catalog/", views.catalog, name="catalog"),
]
