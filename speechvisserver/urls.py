"""speechvisserver URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf import settings
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.contrib import admin
from . import views
from . import data_visualizer
from . import speaker_identification

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^speaker_validation/', views.speaker_validation, name='speaker_validation'),
    url(r'^speaker_identification/', speaker_identification.speaker_identification, name='speaker_identification'),
    url(r'^vocal_categorization/', views.vocal_categorization, name='vocal_categorization'),
    url(r'^data_manager', views.data_manager, name='data_manager'),
    url(r'^data_visualizer', data_visualizer.data_visualizer, name='data_visualizer'),
    url(r'^visualize_feature', data_visualizer.visualize_feature, name='visualize_feature'),
    url(r'^admin/', include(admin.site.urls)),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
