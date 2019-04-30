from django.contrib import admin
from .models import Bridge, BridgeLog, BrokenFlag, ErrorStatus
import smtplib
from email.message import EmailMessage
from calbridgesensors import settings
from django.http import HttpResponse
import csv

# Register your models here.
admin.site.register(BridgeLog)
admin.site.register(BrokenFlag)
admin.site.register(ErrorStatus)


@admin.register(Bridge)
class BridgeAdmin(admin.ModelAdmin):
    list_display = ("name", "status")
    actions = ["take_measurement", "mark_as_repaired","export_as_csv"] # TODO: ADD TEMPLATE VIEW OF TABLES

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
    def export_as_csv(self, request, queryset):
        """https://stackoverflow.com/questions/50952823/django-response-that-contains-a-zip-file-with-multiple-csv-files
        output = StringIO.StringIO()
f = zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED)
f.writestr('first.csv', '<the content of first.csv>')
f.writestr('second.csv', '<the content of second.csv>')
f.writestr('third.csv', '<the content of third.csv>')
f.close()
# Build your response
response = HttpResponse(output.getvalue(), mimetype='application/zip')
response['Content-Disposition'] = 'attachment; filename="yourzipfilename.zip"'
return response"""
        meta = self.model._meta
        field_names = [field.name for field in meta.fields]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename={}.csv'.format(meta)
        writer = csv.writer(response)

        writer.writerow(field_names)
        for obj in queryset:
            row = writer.writerow([getattr(obj, field) for field in field_names])

        return response
    # *******************************************************************************************
    # --------------------------------------------------------------------------------------------
    # *******************************************************************************************