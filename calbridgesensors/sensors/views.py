from django.shortcuts import render
from django.views.generic import DetailView
from django.http import HttpResponse
from django.views.generic.base import TemplateView
from .models import Bridge, BrokenFlag


"""View for home page"""
class SensorsHomeView(TemplateView):
    template_name = "sensors/index.html"

    def get_context_data(self, **kwargs):
        broken_bridges = BrokenFlag.objects.all()




class BridgeView(DetailView):
    model = Bridge
    template_name = "sensors/detail.html"


def bridge_update(request, bridge_name):
    return HttpResponse("hello")