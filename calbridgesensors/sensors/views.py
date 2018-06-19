from django.shortcuts import render
from django.views.generic import TemplateView


"""View for home page"""
def home(request):

    # Check broken bridges
    return render(request, "home.html", context={})


class BridgeView(TemplateView):

    template_name = "bridge.html"