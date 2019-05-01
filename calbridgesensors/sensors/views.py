from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.views.generic.base import TemplateView
from .models import Bridge, BrokenFlag, Reading
from .utils import verify_request, calib_dp_to_di, decimal_rep, parse_db_time
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import json


class SensorsHomeView(LoginRequiredMixin, TemplateView):
    """View for home page"""
    template_name = "sensors/index.html"
    login_url = "/accounts/login/"

    def get_context_data(self, **kwargs):
        broken_bridges = BrokenFlag.objects.all()
        return {"user": self.request.user, "broken_bridges": broken_bridges, "bridges": Bridge.objects.all()}


"""class BridgeView(LoginRequiredMixin, DetailView):
    model = Bridge
    template_name = "sensors/detail.html"
    login_url = "/accounts/login/"
    context_object_name = "bridge"

    def get_context_data(self, **kwargs):
        return {"user": self.request.user}.update(super().get_context_data(**kwargs))"""


@login_required(login_url='/accounts/login/')
def bridge_view(request, pk):
    bridge = get_object_or_404(Bridge, pk=pk)
    rawreadings = bridge.rawreading_set.all()
    len_reading = len(rawreadings)

    calibrateddp = [(0.0, 0.0, None)] * len_reading
    calibratedx = [0.0] * len_reading
    calibratedy = [0.0] * len_reading
    thetas = [0.0] * len_reading
    phis = [0.0] * len_reading
    for i in range(len_reading):
        curr = rawreadings[i]
        dcurr = rawreadings[i].get_reading()
        dx = decimal_rep(calib_dp_to_di(bridge, dcurr.x)) if dcurr.x is not None else None
        dy = decimal_rep(calib_dp_to_di(bridge, dcurr.y)) if dcurr.y is not None else None
        dt = curr.time_taken
        calibrateddp[len_reading - i - 1] = (dx, dy, parse_db_time(dt))
        calibratedx[i] = dx  # First entry would be newest
        calibratedy[i] = dy
        thetas[i] = decimal_rep(dcurr.theta)
        phis[i] = decimal_rep(dcurr.phi)
    calib_json = json.dumps(calibrateddp)
    context = {"user": request.user,
               "damage_recs": bridge.get_damage_records(),
               "repair_recs": bridge.get_repair_records(),
               "bridge": bridge,
               "reading": bridge.latest_reading(),
               "readings": calib_json,
               "readingsx": calibratedx,
               "readingsy": calibratedy,
               "thetas": thetas,
               "phis": phis
               }
    return render(request, "sensors/detail.html", context=context)


@csrf_exempt
def bridge_update(request, pk):
    """JSON"""
    def update_procedure():
        # Connects the front-end request with backend database process
        try:
            br = Bridge.objects.get(pk=pk)
        except Bridge.DoesNotExist:
            return HttpResponse(content='Bridge Unregistered', status=412)
        br.update(request.POST)  # Should resolve all sorts of issues by passing a dictionary in
        return HttpResponse('SUCCESS')

    @csrf_protect
    def protected_update(r, p):
        return update_procedure()

    if verify_request(request.COOKIES):
        return update_procedure()
    else:
        return protected_update(request, pk)










