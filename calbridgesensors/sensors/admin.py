from django.contrib import admin
from .models import Bridge, BridgeLog, BrokenFlag, ErrorStatus, RawReading
import smtplib
from email.message import EmailMessage
from calbridgesensors import settings
from django.http import HttpResponse
from django.utils.timezone import localtime
import csv, zipfile
from io import BytesIO, StringIO
from .utils import parse_db_time, calib_dp_to_di, decimal_rep

# Register your models here.
admin.site.register(BridgeLog)
admin.site.register(BrokenFlag)
admin.site.register(ErrorStatus)


@admin.register(Bridge)
class BridgeAdmin(admin.ModelAdmin):
    list_display = ("name", "status")
    actions = ["take_measurement", "mark_as_repaired","export_reading_csvs"] # TODO: ADD TEMPLATE VIEW OF TABLES

    def status(self, obj):
        return "damaged" if obj.is_broken() else "healthy"

    def take_measurement(self, request, queryset):
        server = smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT)
        server.starttls()
        server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
        msg = EmailMessage()
        msg['From'] = "Bridge Settlement Project Server <{}>".format(settings.EMAIL_HOST_USER)
        msg['To'] = ", ".join([str(obj.number)+settings.TARGET_HOST for obj in queryset])
        msg['Subject'] = "Test Infrastructure"
        response = server.send_message(msg)
        print("Successfully sent")
        server.quit()
        msg = '<html lang="en"><body><script>alert("Success!");</script></body></html>'
        return HttpResponse

    def mark_as_repaired(self, request, queryset):
        for obj in queryset:
            obj.mark_repaired()

    # *******************************************************************************************
    # --- CITE: http://books.agiliq.com/projects/django-admin-cookbook/en/latest/export.html ----
    # *******************************************************************************************
    def export_reading_csvs(self, request, queryset):
        # https://stackoverflow.com/questions/50952823/django-response-that-contains-a-zip-file-with-multiple-csv-files
        import datetime
        print(request.session.get('django_timezone'), datetime.datetime.now().tzinfo)
        if len(queryset) == 0:
            return HttpResponse(status=204)
        else:
            obj = queryset[0]
            fields_names = [field.name for field in RawReading._meta.fields]
            output = BytesIO()
            zf = zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED)
            for obj in queryset:
                csvstring = StringIO()
                cwriter = csv.writer(csvstring)
                cwriter.writerow(['time', 'x', 'calib_x', 'y', 'calib_y', 'z', 'pitch_cam', 'pitch_laser', 'psi',
                                  'counter', 'errors'])
                for rr in obj.rawreading_set.all():
                    row = [None]
                    for f in fields_names:
                        if f in ('id', 'target'):
                            pass
                        elif f in ('x', 'y'):
                            row.append(getattr(rr, f))
                            row.append(decimal_rep(calib_dp_to_di(obj, getattr(rr.get_reading(), f))))
                        elif f == 'time_taken':
                            row[0] = parse_db_time(localtime(getattr(rr, f)))
                        else:
                            row.append(getattr(rr, f))
                    row.append(", ".join([str(es.code) for es in rr.errorstatus_set.all()]))
                    cwriter.writerow(row)
                zf.writestr('{}.csv'.format(obj.name), csvstring.getvalue().encode('utf-8'))
                csvstring.close()
            zf.close()
            response = HttpResponse(output.getvalue(), content_type='application/zip')
            response['Content-Disposition'] = 'attachment; filename={}.zip'.format("readings")
            return response
    # *******************************************************************************************
    # --------------------------------------------------------------------------------------------
    # *******************************************************************************************