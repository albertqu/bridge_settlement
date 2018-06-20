from django.db import models



class Bridge(models.Model):

    name = models.CharField(max_length=40, primary_key=True)
    reading = models.OneToOneField(Reading, on_delete=models.PROTECT, help_text="latest sensor reading of the bridge", blank=True, null=True)
    anomaly = models.BooleanField(default=False)
    broken_times = models.ForeignKey(BridgeLog, on_delete=models.PROTECT)

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

    class Meta:
        verbose_name = "Bridge sensor reading"


class BridgeLog(models.Model):
    DAMAGE = "D"
    REPAIR = "R"
    LOG_TYPES = ((DAMAGE, "Damage Record"),
                 (REPAIR, "Repair Record"))
    log_type = models.CharField(max_length=1, choices=LOG_TYPES)
    log_time = models.DateTimeField(auto_now_add=True, primary_key=True)
    bridge = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="log_time")


class BrokenRecord(models.Model):
     bridge = models.OneToOneField(Bridge, on_delete=models.SET_NULL, to_field="name")