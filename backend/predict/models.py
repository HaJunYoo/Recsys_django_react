# from django.contrib.postgres.fields import ArrayField
from django.db import models
import pandas as pd
# # from sqlalchemy import create_engine
# from django.core.management.base import BaseCommand


class PredResults(models.Model):
    name = models.CharField(max_length=80)
    img = models.CharField(max_length=80)
    review = models.TextField(max_length=256, default='default')
    price = models.CharField(max_length=80, default='default')

    def __str__(self):
        return f'{self.name}'


# class Command(BaseCommand):
#     def handle(self, *args, **options):
#         columns = ['id', 'tag_title', 'tag_price','tag_image_url']
#         df = pd.read_csv('predict/data/product.csv', index_col='id', names=columns, header=0)
#
#         engine = create_engine('sqlite://///Users/jujeonghan/Developer/did_it_walk/db.sqlite3')
#         df.to_sql('모델명'._meta.db_table, if_exists='replace' ,con=engine, index=False)


'''
id
name	main_category	sub_category	brand	
number	tags	price	season	gender	
like	view	sale	coordi	
age18	age19_23	age24_28	age29_33	age34_39	age40	
man	woman	
img	year	only_season	scaled_rating	review

coordi = ArrayField(
    ArrayField(
        models.CharField(max_length=10, blank=True), verbose_name= "coordi",
        size=8,
    ),
    size=8,
)
'''

# 장바구니 모델은 여기를 참조하게 될 것
class Product(models.Model):

    name = models.CharField(verbose_name="name", max_length=255, null=True)
    main_category = models.CharField(verbose_name="main_category", max_length=255, null=True)
    sub_category = models.CharField(verbose_name="sub_category", max_length=255, null=True)
    brand = models.CharField(verbose_name= "brand", max_length=255, null=True)
    number = models.CharField(verbose_name= "number", max_length=255, null=True)
    tags = models.CharField(verbose_name= "tags", max_length=255, null=True)
    price = models.CharField(verbose_name= "price", max_length=255, null=True)
    season = models.CharField(verbose_name= "season", max_length=255, null=True)
    # 리스트 필드
    gender = models.CharField(verbose_name="gender", max_length=255, null=True)

    like = models.FloatField(verbose_name= "like", null=True)
    view = models.FloatField(verbose_name= "view", null=True)
    sale = models.CharField(verbose_name= "sale", max_length=255, null=True)
    # 리스트 필드
    coordi = models.CharField(verbose_name="coordi", max_length=255, null=True)

    age18 = models.FloatField(verbose_name= "age18", null=True)
    age19_23 = models.FloatField(verbose_name= "age19_23", null=True)
    age24_28 = models.FloatField(verbose_name= "age24_28", null=True)
    age29_33 = models.FloatField(verbose_name= "age29_33", null=True)
    age34_39 = models.FloatField(verbose_name= "age34_39", null=True)
    age40 = models.FloatField(verbose_name= "age40", null=True)
    man = models.FloatField(verbose_name= "man", null=True)
    woman = models.FloatField(verbose_name= "woman", null=True)
    img = models.CharField(verbose_name= "img", max_length=255, null=True)
    year = models.CharField(verbose_name= "year", max_length=255, null=True)
    only_season = models.CharField(verbose_name= "only_season", max_length=255, null=True)
    scaled_rating = models.FloatField(verbose_name= "scaled_rating", null=True)
    review = models.CharField(verbose_name = "review", max_length=255, null=True)

    def __str__(self):
        return self.name
