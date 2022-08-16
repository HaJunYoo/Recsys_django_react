from django.urls import path, re_path
from . import views

app_name = "predict"

urlpatterns = [
    # predict_page1 predict_page2
    path('', views.predict_page1, name='prediction_page'),
    path('img/', views.predict_page2, name='image_prediction_page'),

    path('predict/', views.predict, name='prediction'),
    path('img-predict/', views.img_predict, name='image_prediction'),
    # path('item-predict/', views.item_predict, name='item_prediction_page'),

    path('topics/', views.view_topic, name='topics'),

    # path('results/', views.view_results, name='results'),
    path('results/', views.ViewResult.as_view(), name='results'),
    path('product/', views.ViewProduct.as_view(), name='product'),

    path('wordcloud/', views.view_wordcloud, name='wordcloud'),
]


