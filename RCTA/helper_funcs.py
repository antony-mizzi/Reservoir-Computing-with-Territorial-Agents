import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import maps_and_systems as mapnsys
import csv

def log_data(details,fname):
    try:
        df1 = pd.read_csv('logs/'+fname)
        df2 = pd.DataFrame(details, index=[0])
        DF = pd.concat([df1, df2], ignore_index = True)
        #DF.reset_index()
        newframe = DF.copy()
    except:
        #DF = pd.DataFrame(details, index=[0])
        newframe = pd.DataFrame(details, index=[0])
    newframe.to_csv('logs/'+fname, index=False)
    
def log_recon(recon,fname,method='Random',TS='Thomas',size=30):
    recon = [method,TS,size]+recon
    with open('logs/'+fname, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(recon)

def log_homes(homes,fname,method='Random',TS='Thomas'):
    with open('logs/'+fname, 'a', newline='') as file:
        writer = csv.writer(file)
        homes = homes.ravel()
        homes = homes.tolist()
        homes = [method,TS]+homes
        writer.writerow(homes)
        


