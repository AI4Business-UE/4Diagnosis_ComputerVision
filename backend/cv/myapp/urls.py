from django.urls import path
from . import views

urlpatterns = [
    path("select-folder/", views.select_folder, name="select-folder"),
    path("convert/", views.convert, name="convert"),
    path("tiff/<str:job_id>/", views.get_tiff, name="get-tiff"),
    path("api/result-image/<str:job_id>/<str:image_name>/", views.get_result_image),
    path("fibrosis/", views.analyze_fibrosis_degree, name="analyze_fibrosis_degree"),
    path("length/", views.measure_tissue_length, name="measure_tissue_length"),
    # path("glomerule/", views.analyze, name="glomerule"),
]