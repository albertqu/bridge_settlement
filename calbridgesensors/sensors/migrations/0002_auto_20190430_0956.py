# Generated by Django 2.2 on 2019-04-30 16:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sensors', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='errorstatus',
            name='sources',
            field=models.ManyToManyField(blank=True, to='sensors.RawReading'),
        ),
    ]
