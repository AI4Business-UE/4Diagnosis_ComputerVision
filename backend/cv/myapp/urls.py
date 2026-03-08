from django.urls import path
from . import views

urlpatterns = [
    path("select-folder/", views.select_folder, name="select-folder"),
    path("convert/", views.convert, name="convert"),
    path("fibrosis/", views.fibrosis, name="fibrosis"),
    path("length/", views.length, name="length"),
    # path("glomerule/", views.analyze, name="glomerule"),
]