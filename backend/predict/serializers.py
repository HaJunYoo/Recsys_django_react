from rest_framework import serializers
from .models import *

class PredSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredResults
        fields = ['name', 'img', 'review', 'price', 'man', 'woman']
