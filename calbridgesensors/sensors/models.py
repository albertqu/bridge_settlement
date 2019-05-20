from django.db import models
from django.utils import timezone
from .utils import name_validate, max_val, succinct_time_str
from .apps import THRESHOLD_DIS, THRESHOLD_ROT, BUFFER_TIME, BUFFER_MEAS, RECENT_PERIOD


class Bridge(models.Model):
    # Field: reading
    name = models.CharField(max_length=40, primary_key=True)
    init_reading = models.OneToOneField('RawReading', on_delete=models.PROTECT, blank=True, null=True)
    calibration = models.DecimalField(max_digits=10, decimal_places=4, blank=True, default=1.0)
    number = models.BigIntegerField(blank=True, null=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        self.name = name_validate(self.name)
        super(Bridge, self).save()

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
        # TODO: DETERMINE SETTLEMENT RELATIONS
        # ASSUMPTION: QueryDict here contains all the fields as strings
        f = lambda field: eval(data[field]) if field in data else None
        x, y, z, theta, phi, psi, counter = f('x'), f('y'), f('z'), f('theta'), f('phi'), f('psi'), f('counter')
        if counter is None:
            raise RuntimeError("Counter has to be set")
        errors = [eval(e) for e in data['errors'].split(',')] if 'errors' in data and data['errors'] else []
        targets = self.rawreading_set.filter(counter=counter)
        if len(targets) > 0:
            targets[0].add_errors(errors)
            if len(targets) > 1:
                raise RuntimeWarning("Multiple instances with the same counter is present!")
        else:
            new_reading = self.rawreading_set.create(x=x, y=y, z=z, theta=theta, phi=phi, psi=psi, target=self,
                                                     counter=counter)

            if self.init_reading is None and new_reading.is_error_free():
                self.init_reading = new_reading
            latest = new_reading.create_reading()
            self.update_routine(latest)
            self.save()

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
            self.init_reading = None
            self.save()

    def mark_broken(self, damage_rec):
        BrokenFlag.objects.create(bridge=self, damage_record=damage_rec)

    def clean_slate(self):
        rec_set = self.bridgelog_set.all().order_by('-log_time')
        return len(rec_set) == 0 or rec_set[0].log_type == 'R'

    def check_buffer(self):
        latest_damage = self.get_damage_records()[0]
        return latest_damage.time_elapsed() >= BUFFER_TIME \
               or len(self.rawreading_set.filter(time_taken__gte=latest_damage.log_time)) >= BUFFER_MEAS


class RawReading(models.Model):
    # TODO: ADD ABILITY TO RECALIBRATE FOR ALL READINGS
    x = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    y = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    z = models.DecimalField(max_digits=6, decimal_places=2,null=True)
    theta = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    phi = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    psi = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    target = models.ForeignKey(Bridge, on_delete=models.CASCADE, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)
    counter = models.IntegerField(default=-1)

    class Meta:
        verbose_name = "Bridge Sensor Raw Reading"
        ordering = ["target__name", "-time_taken"]

    def __str__(self):
        return "Raw Reading for " + self.target.name + " at " + succinct_time_str(self.time_taken)

    def shows_anomaly(self):
        return not self.is_error_free() or self.get_reading().shows_anomaly()

    def create_reading(self):
        try:
            if self.target.init_reading is None:
                return self.reading_set.create(x=self.x if self.x != -1 else None,
                                               y=self.y if self.y != -1 else None,
                                               z=self.z if self.z != -1 else None,
                                               theta=self.theta, phi=self.phi, psi=self.psi, base=self)
            ix, iy, iz, it, iph, ips = self.target.init_reading.x, self.target.init_reading.y, \
                                       self.target.init_reading.z, self.target.init_reading.theta, \
                                       self.target.init_reading.phi, self.target.init_reading.psi
            if self.disp_valid():
                dx, dy, dz = self.x - ix, self.y - iy, self.z - iz
            else:
                dx, dy, dz = None, None, self.z - iz  # Z is not used here, might be added later
            g = lambda v, iv: (v - iv) if v is not None else None
            dtheta, dphi, dpsi = g(self.theta, it), g(self.phi, iph), g(self.psi, ips)
            return self.reading_set.create(x=dx, y=dy, z=dz, theta=dtheta, phi=dphi, psi=dpsi, base=self)
        except:
            return self.reading_set.create(base=self)

    def get_reading(self):
        rmanager = self.reading_set
        if len(rmanager.all()) == 0:
            return self.create_reading()
        else:
            return self.reading_set.latest('time_taken')

    def disp_valid(self):
        return self.x is not None and self.y is not None and self.z is not None \
               and self.x >=0 and self.y >=0 and self.z >= 0

    def is_error_free(self):
        return self.disp_valid() and self.phi is not None and self.theta is not None and self.psi is not None

    def add_errors(self, errors):
        for e in errors:
            try:
                err = ErrorStatus.objects.get(pk=e)
            except:
                err = ErrorStatus.objects.create(code=e, status="")
            self.errorstatus_set.add(err)
            err.save()
        self.save()


class Reading(models.Model):
    # TODO: ADD ABILITY TO RECALIBRATE FOR ALL READINGS
    x = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    y = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    z = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    theta = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    phi = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    psi = models.DecimalField(max_digits=6, decimal_places=2, blank=True, null=True)
    base = models.ForeignKey('RawReading', on_delete=models.PROTECT, null=True, unique_for_date="time_taken")
    time_taken = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Bridge Sensor Reading"
        ordering = ["base__target__name", "-time_taken"]

    def __str__(self):
        return "Reading for " + self.base.target.name + " at " + succinct_time_str(self.time_taken)

    def shows_anomaly(self):
        return self.has_invalid() or max_val(self.x, self.y, self.z) > THRESHOLD_DIS \
               or max_val(self.theta, self.phi, self.psi) > THRESHOLD_ROT

    def has_invalid(self):
        return self.x is None or self.y is None or self.z is None \
               or self.theta is None or self.phi is None or self.psi is None

    def get_errors(self):
        return self.base.error_set.all()


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


class ErrorStatus(models.Model):
    code = models.IntegerField(primary_key=True)
    status = models.CharField(max_length=80, default="")
    sources = models.ManyToManyField(RawReading, blank=True)

    class Meta:
        verbose_name = "Error Status Code"
        ordering = ["code"]

    def __str__(self):
        return "Code {}, {}".format(self.code, self.status)



