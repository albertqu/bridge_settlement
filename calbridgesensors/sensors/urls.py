from django.urls import path, re_path, register_converter
from .converters import BridgeNameConverter
from . import views

register_converter(BridgeNameConverter, 'brn')

urlpatterns = [
    path("", views.SensorsHomeView, "sensors"),
    path("<brn:pk>/", views.BridgeView.as_view(), "detail"),
    path("<brn:pk>/update/", views.bridge_update)
]