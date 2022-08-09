# Generated by Django 4.0.3 on 2022-08-09 12:32

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PredResults',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=80)),
                ('img', models.CharField(max_length=80)),
                ('review', models.TextField(default='default', max_length=256)),
                ('price', models.CharField(default='default', max_length=80)),
            ],
        ),
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, null=True, verbose_name='name')),
                ('main_category', models.CharField(max_length=255, null=True, verbose_name='main_category')),
                ('sub_category', models.CharField(max_length=255, null=True, verbose_name='sub_category')),
                ('brand', models.CharField(max_length=255, null=True, verbose_name='brand')),
                ('number', models.CharField(max_length=255, null=True, verbose_name='number')),
                ('tags', models.CharField(max_length=255, null=True, verbose_name='tags')),
                ('price', models.CharField(max_length=255, null=True, verbose_name='price')),
                ('season', models.CharField(max_length=255, null=True, verbose_name='season')),
                ('gender', models.CharField(max_length=255, null=True, verbose_name='gender')),
                ('like', models.FloatField(null=True, verbose_name='like')),
                ('view', models.FloatField(null=True, verbose_name='view')),
                ('sale', models.CharField(max_length=255, null=True, verbose_name='sale')),
                ('coordi', models.CharField(max_length=255, null=True, verbose_name='coordi')),
                ('age18', models.FloatField(null=True, verbose_name='age18')),
                ('age19_23', models.FloatField(null=True, verbose_name='age19_23')),
                ('age24_28', models.FloatField(null=True, verbose_name='age24_28')),
                ('age29_33', models.FloatField(null=True, verbose_name='age29_33')),
                ('age34_39', models.FloatField(null=True, verbose_name='age34_39')),
                ('age40', models.FloatField(null=True, verbose_name='age40')),
                ('man', models.FloatField(null=True, verbose_name='man')),
                ('woman', models.FloatField(null=True, verbose_name='woman')),
                ('img', models.CharField(max_length=255, null=True, verbose_name='img')),
                ('year', models.CharField(max_length=255, null=True, verbose_name='year')),
                ('only_season', models.CharField(max_length=255, null=True, verbose_name='only_season')),
                ('scaled_rating', models.FloatField(null=True, verbose_name='scaled_rating')),
                ('review', models.CharField(max_length=255, null=True, verbose_name='review')),
            ],
        ),
    ]