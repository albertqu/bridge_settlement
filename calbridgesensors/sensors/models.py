from django.db import models
from .utils import name_validate
from django.utils import timezone


class Bridge(models.Model):
    # Field: reading
    name = models.CharField(max_length=40, primary_key=True)

    class Meta:
        ordering = ["name"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name_validate(self.name)

    def __str__(self):
        return self.name

    def latest_reading(self):
        return self.reading_set.all()[0]

    def get_damage_records(self):
        return self.bridgelog_set.filter(log_type="D")

    def get_repair_records(self):
        return self.bridgelog_set.filter(log_type="R")

    def is_broken(self):
        return self.brokenflag

    def add_reading(self, x, y, z, theta, phi, psi):
        return self.reading_set.create(x=x, y=y, z=z, theta=theta, phi=phi, psi=psi, bridge=self)

    def update(self, x, y, z, theta, phi, psi):
        old_reading = self.latest_reading()
        new_reading = self.add_reading(x, y, z, theta, phi, psi)

    def mark_broken(self, bridge_log):
        self.brokenflag = BrokenFlag(bridge=self, first_broken_record=bridge_log)


class Reading(models.Model):
    x = models.DecimalField(max_digits=4, decimal_places=2)
    y = models.DecimalField(max_digits=4, decimal_places=2)
    z = models.DecimalField(max_digits=4, decimal_places=2)
    theta = models.DecimalField(max_digits=4, decimal_places=2)
    phi = models.DecimalField(max_digits=4, decimal_places=2)
    psi = models.DecimalField(max_digits=4, decimal_places=2)
    bridge = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Bridge Sensor Reading"
        ordering = ["bridge__name", "-time_taken"]

    def __str__(self):
        return "Reading for " + self.bridge.name


class BridgeLog(models.Model):
    DAMAGE = "D"
    REPAIR = "R"
    LOG_TYPES = ((DAMAGE, "Damage Record"),
                 (REPAIR, "Repair Record"))
    log_type = models.CharField(max_length=1, choices=LOG_TYPES)
    log_time = models.DateTimeField(auto_now_add=True, primary_key=True)
    bridge = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="log_time")

    class Meta:
        ordering = ["bridge__name", "log_type", "-log_time"]

    def __str__(self):
        return self.get_log_type_display() + " for " + self.bridge.name

    def description(self):
        return self.get_log_type_display() + " at " + self.log_time


class BrokenFlag(models.Model):
    bridge = models.OneToOneField(Bridge, on_delete=models.CASCADE)
    first_broken_record = models.OneToOneField(BridgeLog, on_delete=models.PROTECT)

    class Meta:
        ordering = ["bridge__name"]

    def broken_time(self):
        return timezone.now() - self.first_broken_record.log_time
