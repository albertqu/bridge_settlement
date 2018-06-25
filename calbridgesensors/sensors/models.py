from django.db import models
import re


class Bridge(models.Model):
    name = models.CharField(max_length=40, primary_key=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target = r'[^ A-Za-z0-9]'
        self.name = str(re.sub(target, " ", self.name)).title()

    def __str__(self):
        return self.name

    def latest_reading(self):
        return self.reading_set.all()[0]

    def get_damage_records(self):
        return self.bridgelog_set.filter(log_type="D")

    class Meta:
        ordering = ["name"]


class Reading(models.Model):
    x = models.DecimalField(max_digits=4, decimal_places=2)
    y = models.DecimalField(max_digits=4, decimal_places=2)
    z = models.DecimalField(max_digits=4, decimal_places=2)
    theta = models.DecimalField(max_digits=4, decimal_places=2)
    phi = models.DecimalField(max_digits=4, decimal_places=2)
    psi = models.DecimalField(max_digits=4, decimal_places=2)
    bridge = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "Reading for " + self.bridge.name

    class Meta:
        verbose_name = "Bridge Sensor Reading"
        ordering = ["bridge__name", "-time_taken"]


class BridgeLog(models.Model):
    DAMAGE = "D"
    REPAIR = "R"
    LOG_TYPES = ((DAMAGE, "Damage Record"),
                 (REPAIR, "Repair Record"))
    log_type = models.CharField(max_length=1, choices=LOG_TYPES)
    log_time = models.DateTimeField(auto_now_add=True, primary_key=True)
    bridge = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="log_time")

    class Meta:
        ordering = ["-log_time", "log_type", "bridge__name"]


class BrokenFlag(models.Model):
    bridge = models.OneToOneField(Bridge, on_delete=models.CASCADE)
    record = models.OneToOneField(BridgeLog, on_delete=models.PROTECT)

    class Meta:
        ordering = ["bridge__name"]