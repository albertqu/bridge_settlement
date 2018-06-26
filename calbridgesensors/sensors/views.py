from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView
from django.http import HttpResponse
from django.views.generic.base import TemplateView
from .models import Bridge, BrokenFlag


class SensorsHomeView(LoginRequiredMixin, TemplateView):
    """View for home page"""
    template_name = "sensors/index.html"
    login_url = "/accounts/login/"

    def get_context_data(self, **kwargs):
        broken_bridges = BrokenFlag.objects.all()
        return {"user": self.request.user, "broken_bridges": broken_bridges, "bridges": Bridge.objects.all()}


class BridgeView(LoginRequiredMixin, DetailView):
    model = Bridge
    template_name = "sensors/detail.html"
    login_url = "/accounts/login/"
    context_object_name = "bridge"

    def get_context_data(self, **kwargs):
        return {"user", self.request.user}.update(super().get_context_data(**kwargs))


def bridge_update(request, bridge_name):
    return HttpResponse("hello")