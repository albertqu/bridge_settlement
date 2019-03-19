from django.db import models
from django.utils import timezone
from .utils import name_validate, max_val, succinct_time_str
from .apps import THRESHOLD_DIS, THRESHOLD_ROT, BUFFER_TIME, BUFFER_MEAS, RECENT_PERIOD


class Bridge(models.Model):
    # Field: reading
    name = models.CharField(max_length=40, primary_key=True)
    init_reading = models.OneToOneField('RawReading', on_delete=models.PROTECT, blank=True, null=True)

    class Meta:
        ordering = ["name"]

    def __init__(self, *args, **kwargs):
        super(Bridge, self).__init__(*args, **kwargs)
        self.name = name_validate(self.name)

    def __str__(self):
        return self.name

    def latest_reading(self):
        r_set = self.rawreading_set.all()
        return r_set[0].get_reading() if r_set else None

    def latest_rawreading(self):
        r_set = self.rawreading_set.all()
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
        new_reading = self.rawreading_set.create(x=float(data['x']), y=float(data['y']), z=float(data['z']),
                                              theta=float(data['theta']), phi=float(data['phi']),
                                              psi=float(data['psi']), target=self)
        latest = new_reading.create_reading()
        self.update_routine(latest)

    def update_routine(self, new_reading):
        if new_reading.shows_anomaly():
            # Check whether it has a broken flag, whether it is new anomaly
            if not self.is_broken():
                # Means it's new. Checks latest damage record, don't raise alarm if it is within buffer time
                if self.clean_slate():
                    self.bridgelog_set.create(log_type='D', bridge=self)
                elif self.check_buffer():
                    # Already over the buffer time, time to raise alarm
                    self.mark_broken(self.get_damage_records()[0])
                # Else it is within the buffer time, wait for sometime
            # Else it's old. No need to create new records. Also no need to raise alarm again
        else:
            # Now it seems normal. Three scenerios: 1) Repaired, 2) False Alarm Previously, 3) Good In the First Place
            if self.is_broken() or not self.clean_slate():
                self.mark_repaired()

    def refresh(self):
        self.update_routine(self.latest_reading())

    def mark_repaired(self):
        # TODO: ADD AUTHENTICATION IN THE VIEW
        self.bridgelog_set.create(log_type='R', bridge=self)
        if self.is_broken():
            self.brokenflag.delete()

    def mark_broken(self, damage_rec):
        BrokenFlag.objects.create(bridge=self, damage_record=damage_rec)

    def clean_slate(self):
        rec_set = self.bridgelog_set.all().order_by('-log_time')
        return len(rec_set) == 0 or rec_set[0].log_type == 'R'

    def check_buffer(self):
        latest_damage = self.get_damage_records()[0]
        return latest_damage.time_elapsed() >= BUFFER_TIME \
               or len(self.reading_set.filter(time_taken__gte=latest_damage.logtime)) >= BUFFER_MEAS


class RawReading(models.Model):
    # TODO: ADD ABILITY TO RECALIBRATE FOR ALL READINGS
    x = models.DecimalField(max_digits=4, decimal_places=2)
    y = models.DecimalField(max_digits=4, decimal_places=2)
    z = models.DecimalField(max_digits=4, decimal_places=2)
    theta = models.DecimalField(max_digits=4, decimal_places=2)
    phi = models.DecimalField(max_digits=4, decimal_places=2)
    psi = models.DecimalField(max_digits=4, decimal_places=2)
    target = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Bridge Sensor Raw Reading"
        ordering = ["bridge__name", "-time_taken"]

    def __str__(self):
        return "Raw Reading for " + self.bridge.name + " at " + succinct_time_str(self.time_taken)

    def shows_anomaly(self):
        return self.get_reading().shows_anomaly()

    def create_reading(self):
        ix, iy, iz, it, iph, ips = self.target.init_reading.x, self.target.init_reading.y, \
                                   self.target.init_reading.z, self.target.init_reading.theta, \
                                   self.target.init_reading.phi, self.target.init_reading.psi
        return self.reading_set.create(x=self.x - ix, y=self.y - iy, z=self.z - iz, theta=self.theta - it,
                                       phi=self.phi - iph, psi=self.psi - ips, base=self)

    def get_reading(self):
        return self.reading_set.all()[0]


class Reading(models.Model):
    # TODO: ADD ABILITY TO RECALIBRATE FOR ALL READINGS
    x = models.DecimalField(max_digits=4, decimal_places=2)
    y = models.DecimalField(max_digits=4, decimal_places=2)
    z = models.DecimalField(max_digits=4, decimal_places=2)
    theta = models.DecimalField(max_digits=4, decimal_places=2)
    phi = models.DecimalField(max_digits=4, decimal_places=2)
    psi = models.DecimalField(max_digits=4, decimal_places=2)
    base = models.ForeignKey('RawReading', on_delete=models.PROTECT, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Bridge Sensor Reading"
        ordering = ["base__bridge__name", "-time_taken"]

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
