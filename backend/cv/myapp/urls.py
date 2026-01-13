from django.urls import path
from . import views


print("🔥 myapp.urls LOADED 🔥")

urlpatterns = [
    path("import_slide/", views.import_slide, name="import_slide"),
    path("convert/", views.convert, name="convert"),
    path("analyze/", views.analyze, name="analyze"),
]