from django.test import TestCase
from .models import Bridge, Reading

# Create your tests here.
def stage1():
    bridge = Bridge.objects.get(pk='Golden Gate Bridge')
    print(bridge.name)
    assert not bridge.is_broken()
    data0 = {'x': 240, 'y': 300, 'z': 0, 'theta': 0, 'phi': 0, 'psi': 0}
    bridge.update(data0)
    


