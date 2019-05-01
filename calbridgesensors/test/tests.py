from django.test import TestCase
from sensors.models import Bridge, Reading

# Create your tests here.
def stage1():
    bridge = Bridge.objects.get(pk='Golden Gate Bridge')
    print(bridge.name)
    assert not bridge.is_broken()
    data0 = {'x': 240, 'y': 300, 'z': 0, 'theta': 0, 'phi': 0, 'psi': 0}
    bridge.update(data0)

curr=1
def stage2():
    global curr
    bridge = Bridge.objects.all()[0]
    data0 = {'x':"-1", 'y':"-1", 'z':'0', 'theta':'0', 'phi':'0', 'psi': '0', 'counter': str(curr), 'errors': "1,2"}
    curr += 1
    bridge.update(data0)
    data0['errors'] = ''
    data0['counter'] = str(curr)
    curr += 1
    bridge.update(data0)
    data1 = {'x': '240', 'y': '300', 'z': '0', 'theta':'0', 'phi':'0', 'psi': '0', 'counter': str(curr)}
    bridge.update(data1)
    curr += 1
    data0['counter'] = str(curr)
    bridge.update(data1)
    curr += 1
    data0['counter'] = str(curr)
    bridge.update(data0)
