# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-08-27 05:28
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('checker', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Check',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images')),
                ('type', models.CharField(blank=True, max_length=50, null=True)),
                ('percentage', models.IntegerField(blank=True, null=True)),
                ('comments', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='checker.Comment')),
            ],
        ),
        migrations.RemoveField(
            model_name='upload',
            name='comments',
        ),
        migrations.DeleteModel(
            name='Upload',
        ),
    ]
