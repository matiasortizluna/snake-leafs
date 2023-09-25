from google.colab import drive
import os
import pandas as pd

#Mount file from Google drive
drive.mount('/content/gdrive/', force_remount=True)

#Acess Google drive
%cd gdrive/MyDrive

#Get some info about paths
list_files = []
print(os.listdir())
#print(os.getcwd())