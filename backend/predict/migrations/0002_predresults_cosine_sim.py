# Generated by Django 3.2.12 on 2022-08-16 09:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='predresults',
            name='cosine_sim',
            field=models.FloatField(default=0.0),
        ),
    ]