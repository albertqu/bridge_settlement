from django.urls import path, re_path, register_converter
from .converters import BridgeNameConverter
from . import views

register_converter(BridgeNameConverter, 'brn')

app_name = "sensors"

urlpatterns = [
    path('', views.SensorsHomeView.as_view(), name='index'),
    path('<brn:pk>/', views.bridge_view, name='detail'),
    #re_path(r'^(?P<pk>[\w-]+)/$', views.BridgeView.as_view(), name='detail'),
    path('<brn:pk>/update/', views.bridge_update)
]
