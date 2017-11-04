# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:16:59 2017

@author: Meteor
"""

### Import Library
import requests
from bs4 import BeautifulSoup
import os
import csv
import pandas as pd
import platform

### System Config
if platform.system() == 'Windows':
    print('__Windows__')
    path = r'C:\\Users\\Meteor\\Desktop\\'
else:
    print(platform.system())
    path = r'/Users/meteorchang/LocalDesktop/01.python/'
    
print(path) 

"""
# Mac Path = r'/Users/meteorchang/LocalDesktop/01.python//'
# Mac Github = r'/Users/meteorchang/Documents/GitHub/DevelopingFunction/'
# Win Path = r'C:\\Users\\Meteor\\Desktop\\'
"""

### Set program path & Read local script
import os
import sys
code_path = r'C:\\Users\\Meteor\\Desktop\\Stock\\PythonCode\\package'
os.chdir(code_path)
sys.path.append(r'C:\\Users\\Meteor\\Desktop\\Stock\\PythonCode\\package')


### Read files within folders
file_list = []
raw_path = r'C:\Users\Meteor\Desktop\Stock\Parserd\每日收盤行情_除權證\證券代號'
for dirPath, dirNames, fileNames in os.walk(raw_path):
   for f in fileNames:
        file_list.append(os.path.join(dirPath, f))  








