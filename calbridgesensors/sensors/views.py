from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic import DetailView
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
    readings = bridge.rawreading_set.all()
    len_reading = len(readings)
    ground_zero = readings[len_reading - 1]
    ground_zero_x = ground_zero.x
    ground_zero_y = ground_zero.y
    calibrated = [(0.0, 0.0, None)] * len_reading
    calibratedx = [0.0] * len_reading
    calibratedy = [0.0] * len_reading
    dates = [None] * len_reading
    for i in range(len_reading):
        curr = readings[i]
        dx = decimal_rep(calib_dp_to_di(curr.x - ground_zero_x))
        dy = decimal_rep(calib_dp_to_di(curr.y - ground_zero_y))
        dt = curr.time_taken
        calibrated[len_reading - i - 1] = (dx, dy, parse_db_time(dt))
        calibratedx[i] = dx
        calibratedy[i] = dy
    calib_json = json.dumps(calibrated)
    #print(calibrated[-10:])
    #json_readings = json.dumps(readings)
    context = {"user": request.user,
               "damage_recs": bridge.get_damage_records(),
               "repair_recs": bridge.get_repair_records(),
               "bridge": bridge,
               "reading": bridge.latest_reading(),
               "readings": calib_json,
               "readingsx": calibratedx,
               "readingsy": calibratedy
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










