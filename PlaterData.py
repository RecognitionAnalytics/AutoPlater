
import threading
import random
import time
from queue import Queue, Empty
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
import re
import requests
import json
from datetime import datetime
import win32pipe, win32file, pywintypes

class Grapher_Send:
    def __init__(self):
        print("Pipe server")
        
        self.namepipe = win32pipe.CreateNamedPipe(
                r'\\.\pipe\PData',
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, 65536, 65536,
                0,
                None)
        print('Wait for connection')
        win32pipe.ConnectNamedPipe(self.namepipe, None)
        print('Connected')
        
    def SendGraph(self,graph,gtype,xdata,ydata):
        try:
            x_bytes = np.array( xdata,dtype=float).tobytes()
            y_bytes = np.array( ydata,dtype=float).tobytes()
            
            header =(f"{graph},{gtype},{len(x_bytes)},{len(y_bytes)}").ljust(100, ' ')
            print(header+"'")
            
            name_bytes = str.encode(header,'utf-8')
             
            name_bytes=b"".join([name_bytes,x_bytes])
            name_bytes=b"".join([name_bytes,y_bytes])
            
            win32file.WriteFile(self.namepipe, name_bytes)

        except Exception as ex:
            print(ex)
            print('No Server')
            

    def Close(self):
        win32file.CloseHandle(self.namepipe)
        
    
class ConductanceDatabaseObject:
    def __init__(self, wafer, chip):
        self.wafer=wafer
        self.chip=chip
        self.folder =f'C:/Data/Plater/{wafer}/{chip}'
        self.filename = f'{self.folder}/conductance.json'
        self.dformat ='%Y-%m%d-%H%M%S'
        
        if os.path.exists(f'C:/Data/Plater/{wafer}')==False:
            os.mkdir(f'C:/Data/Plater/{wafer}')
        if os.path.exists(self.folder)==False:    
            os.mkdir(self.folder)
        
        if os.path.exists(self.filename):
            print('Found existing conductance database')
            with open(self.filename,'r') as f:
                jData= f.read()
            if jData.strip()=='':
                self.conductanceDatabase={}
            else:
                self.conductanceDatabase=json.loads(jData)
        else:
            self.conductanceDatabase={}
            
    def __getitem__(self, stage):
        return self.conductanceDatabase[stage]

    def SetValue(self, stage,channel, value):
        if self.conductanceDatabase.get(stage) is None:
            self.conductanceDatabase[stage]={}
            
        self.conductanceDatabase[stage][channel] = value
 
        #write out the parameters for this channel for later checks
        with open(self.filename, 'w') as f:
            f.write(json.dumps(self.conductanceDatabase, indent = 4) )

    def SavePlate(self,stage, channel, infos, attempt, currents, sideApplied, plater):
        now = datetime.now()
        filename=f'{self.folder}/{now.strftime(self.dformat)}_{channel}_{stage}_{attempt}_plate_RT.npy'
        with open(filename, 'wb') as f:
            np.save(f, json.dumps(infos, indent = 4))
            np.save(f, [plater .samplesPerSec])
            np.save(f, currents)
            np.save(f, sideApplied)
        return filename
            
    def LoadPlate(self, stage,channel):
        files = os.listdir(self.folder)
        plateFiles = [x for x in files if (channel in x) and (stage in x) and ('_plate_RT.' in x)  ]
        plateFiles.sort()
        currents=[]
        sides=[]
        for file in plateFiles:
            with open(f'{self.folder}/{file}', 'rb') as f:
                infos = json.loads( str(np.load(f)))
                timeStep = np.load(f)[0]
                current =np.load(f)
                sideApplied = np.load(f)
                currents.append(current)
                sides.append(sideApplied)
                
                
        currents = np.concatenate(  currents )
        sides = np.concatenate(  sides )
        return infos,  timeStep,    currents ,sides
    
    def SaveIV(self, stage, channel, bias, currents,conductance_nA):
        now = datetime.now()
        filename=f'{self.folder}/{now.strftime(self.dformat)}_{channel}_{stage}_IV.npy'
        with open(filename, 'wb') as f:
            np.save(f, conductance_nA)
            np.save(f, bias)
            np.save(f, currents)   
        return filename
    
    def SendInfo(self,info_JSON):
        r = requests.post('https://10.212.27.176:7006/AddDataPoint', json=info_JSON,verify=False)
        print(r)
        
    def PlotHistory(self):
        channelValues ={}
        for trial in self.conductanceDatabase:
            channelConds =self. conductanceDatabase[trial]
            for channel in channelConds:
                if channel not in channelValues:
                    channelValues[channel]=[]
                cond = channelConds[channel]
                channelValues[channel].append([trial,cond])

        plt.figure(figsize=(15,5))        
        for channel in channelValues:
            try:
                values=channelValues[channel]
                plt.semilogy( [x[0] for x in values],[x[1] for x in values],label=channel)
            except:
                pass
        plt.ylabel('Conductance (nS)')
        plt.show()        
        
    def PlotHistoryLin(self):
        channelValues ={}
        for trial in self.conductanceDatabase:
            channelConds =self. conductanceDatabase[trial]
            for channel in channelConds:
                if channel not in channelValues:
                    channelValues[channel]=[]
                cond = channelConds[channel]
                channelValues[channel].append([trial,cond])

        plt.figure(figsize=(15,5))        
        maxx=0
        for channel in channelValues:
            try:
                values=channelValues[channel]
                y=[x[1] for x in values]
                maxx=np.max([maxx,np.max(y)])
                plt.plot( [x[0] for x in values],y,label=channel)
            except:
                pass
        plt.ylim([0, np.min([100,maxx])])
        plt.ylabel('Conductance (nS)')
        plt.show()          