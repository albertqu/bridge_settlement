from django.contrib import admin
from .models import Bridge, BridgeLog, BrokenFlag

# Register your models here.
admin.site.register(Bridge)
admin.site.register(BridgeLog)
admin.site.register(BrokenFlag)