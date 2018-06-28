from django.db import models
from django.utils import timezone
from .utils import name_validate, max_val, succinct_time_str
from .apps import THRESHOLD_DIS, THRESHOLD_ROT, BUFFER_TIME, RECENT_PERIOD


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
        r_set = self.reading_set.all()
        return r_set[0] if r_set else None

    def get_damage_records(self):
        return self.bridgelog_set.filter(log_type="D")

    def get_repair_records(self):
        return self.bridgelog_set.filter(log_type="R")

    def is_broken(self):
        try:
            return self.brokenflag
        except AttributeError:
            return False

    def update(self, data):
        """ Creates new reading, checks anomaly;
        --> Raises Broken Flag if showing anomaly for longer than BUFFER_TIME;
        --> Marks bridge as repaired when repaired. """
        new_reading = self.reading_set.create(x=float(data['x']), y=float(data['y']), z=float(data['z']),
                                              theta=float(data['theta']), phi=float(data['phi']),
                                              psi=float(data['psi']), bridge=self)
        if new_reading.shows_anomaly():
            # Check whether it has a broken flag, whether it is new anomaly
            if not self.is_broken():
                # Means it's new. Checks latest damage record, don't raise alarm if it is within buffer time
                damage_recs = self.get_damage_records()
                if damage_recs:
                    latest_damage = damage_recs[0]
                    offset = latest_damage.time_elapsed().days
                    if offset > BUFFER_TIME:
                        # Latest Damage Record not created yet
                        self.bridgelog_set.create(log_type='D', bridge=self)
                    elif offset == BUFFER_TIME:
                        # Already over the buffer time, time to raise alarm
                        self.mark_broken(latest_damage)
                    # Else it is within the buffer time, wait for sometime
                else:
                    # No Damage Record Yet
                    self.bridgelog_set.create(log_type='D', bridge=self)
            # Else it's old. No need to create new records. Also no need to raise alarm again
        else:
            # Now it seems normal. Three scenerios: 1) Repaired, 2) False Alarm Previously, 3) Good In the First Place
            if self.is_broken():
                self.mark_repaired()

    def mark_repaired(self):
        self.bridgelog_set.create(log_type='R', bridge=self)
        self.brokenflag.delete()

    def mark_broken(self, damage_rec):
        BrokenFlag.objects.create(bridge=self, broken_record=damage_rec)


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
        return "Reading for " + self.bridge.name + " at " + succinct_time_str(self.time_taken)

    def shows_anomaly(self):
        return max_val(self.x, self.y, self.z) > THRESHOLD_DIS \
               or max_val(self.theta, self.phi, self.psi) > THRESHOLD_ROT


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
        return self.get_log_type_display() + " for " + self.bridge.name + " at " + succinct_time_str(self.log_time)

    def description(self):
        return self.get_log_type_display() + " at " + str(self.log_time)

    def time_elapsed(self):
        return timezone.now() - self.log_time

    def is_recent(self):
        return self.time_elapsed() <= RECENT_PERIOD


class BrokenFlag(models.Model):
    bridge = models.OneToOneField(Bridge, on_delete=models.CASCADE)
    damage_record = models.OneToOneField(BridgeLog, on_delete=models.PROTECT)

    class Meta:
        ordering = ["bridge__name"]

    def broken_time(self):
        return self.damage_record.time_elapsed()
