# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Annotation',
            fields=[
                ('id', models.AutoField(verbose_name='ID', auto_created=True, primary_key=True, serialize=False)),
                ('speaker', models.CharField(max_length=50)),
                ('category', models.CharField(null=True, blank=True, max_length=50)),
                ('transcription', models.CharField(null=True, blank=True, max_length=400)),
                ('sensitive', models.BooleanField(default=True)),
                ('coder', models.CharField(max_length=50)),
                ('method', models.CharField(null=True, blank=True, max_length=50)),
                ('annotated', models.DateTimeField(null=True, auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Child',
            fields=[
                ('id', models.CharField(serialize=False, primary_key=True, max_length=50)),
                ('key', models.CharField(max_length=50)),
                ('dob', models.DateField()),
                ('gender', models.CharField(max_length=1)),
            ],
        ),
        migrations.CreateModel(
            name='Recording',
            fields=[
                ('id', models.CharField(serialize=False, primary_key=True, max_length=50)),
                ('directory', models.CharField(null=True, max_length=200)),
                ('child', models.ForeignKey(to='speechvisserver.Child')),
            ],
        ),
        migrations.CreateModel(
            name='Segment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', auto_created=True, primary_key=True, serialize=False)),
                ('number', models.IntegerField()),
                ('start', models.DecimalField(max_digits=10, decimal_places=4)),
                ('end', models.DecimalField(max_digits=10, decimal_places=4)),
                ('filename', models.CharField(blank=True, max_length=400)),
                ('recording', models.ForeignKey(to='speechvisserver.Recording')),
            ],
        ),
        migrations.AddField(
            model_name='annotation',
            name='segment',
            field=models.ForeignKey(related_name='segment', to='speechvisserver.Segment'),
        ),
        migrations.AlterUniqueTogether(
            name='segment',
            unique_together=set([('recording', 'number')]),
        ),
    ]
