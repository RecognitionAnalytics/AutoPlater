#!python -m pip install nidaqmx
import threading
import random
import time
from queue import Queue, Empty
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import LineGrouping
from nidaqmx.constants import AcquisitionType 
from nidaqmx.stream_readers import AnalogMultiChannelReader,AnalogSingleChannelReader
import os
from IPython.display import clear_output
from nidaqmx.constants import WAIT_INFINITELY
import re
import requests
import datetime
import json

    
#with Plater(10000) as plater:    
        #conductance_nA, outBias,currents=plater.runIV2( maxVoltage_mV=25, slew_mV_s=100)
        #conductance_nA, outBias,currents=plater.pulseBias( maxVoltage_mV=1000, pulseLengthS = .1,totalLengthS=.3)
        #plater.ResetDevice()
        #conductance_nA_IV,pulseConductance,currents=plater.runPulseTrain(maxVoltage_mV=1000,
        #                                                                 pulseLengthS = .1,
        #                                                                 totalLengthS=.3,
        #                                                                 checkVoltage_mV=100,
        #                                                                 numberPulses=10, plot=True )
        #conductance_nA, current = plater.runRT(voltage_mV=1000, time_s=3)
        #plater.setBias(0)
        #plater.TopElectrode()
        #plater.ResetDevice()
        #tripped, currents = plater. runConstantBias2(maxVoltage_mV=1000, threshold_nA=5, maxTime_S=8,plot=True)
        #print(tripped)

deviceChannels=[]
for chan in ['N','W']:
    for index in range(8,0,-1):
        deviceChannels.append(chan+str(index))
for chan in ['E','S']:
    for index in range(1,9):
        deviceChannels.append(chan+str(index))
        
def PlatingLoop(stage,plater, setParameters,platingChannels,repeatChannels,conductanceDatabase):
       
    while len(platingChannels)>0:
        channel=  platingChannels.pop(0)
        infos = setParameters(channel)

        print(channel)

        #run an IV to check if the channel is shorted before trying to plate
        plater.SelectChannel(channel)
        conductance_nA=RunIV(infos, channel,stage+ '_Before', 100, plater, conductanceDatabase)
        plater.setBias(0)

        if conductance_nA>infos['shortedThreshold_nS']:
                print('Shorted' + channel)   
                continue

        time.sleep(.5)
        attempt =0
        start = time.time()

        while  conductance_nA<infos['finishThreshold_nA'] and attempt<infos['platingAttempts'] :

            if (attempt==0 and infos['quickPulse']):#
                conductance_nA_IV,_,curr=plater.runPulseTrain(maxVoltage_mV=infos['bias_V']*1000.0, pulseLengthS=.05,
                                                        totalLengthS=.1, checkVoltage_mV=100,
                                                        numberPulses=3, plot=True)                
                plater.ResetDevice()

            tripped,currents=Plate(infos,stage,channel,attempt,conductanceDatabase,plater)
            plater.setBias(0)

            plater.TopElectrode()
            plater.setThreshold(current_nA = 200)
            plater.disableThreshold()


            if infos['redoAfterIV']==False:
                conductance_nA=RunIV(infos, channel,stage, 100, plater, conductanceDatabase, slew_mV_s=100)
                break
            else:
                #higher voltages help avoid the elbow from these tunnel junctions from hiding a small gap
                conductance_nA=RunIV(infos, channel,stage, 300, plater, conductanceDatabase, slew_mV_s=300)


            print(channel)
            print('attempt',attempt)

            if tripped ==False:
                break

            if np.max(currents)>50:
                break

            if (time.time()-start)>infos['maxTime_Min']*10:
                repeatChannels.append(channel)
                break


            attempt+=1
        plater.setBias(0)                     
     
        
def RunIV(infos, channel, stage,  maxVoltage_mV, plater, conductanceDatabase,  slew_mV_s=300, topElectrode=True):
    if topElectrode:
        plater.TopElectrode()
        electrode='Top'
    else:
        plater.BottomElectrode()
        electrode='Bottom'
        
    plater.setThreshold(current_nA = 200)
    plater.ResetDevice()
    plater.disableThreshold()
    conductance_nA, outBias,currents =plater.runIV2( maxVoltage_mV=maxVoltage_mV, slew_mV_s=slew_mV_s, plot=True)
    
    

    conductanceDatabase.SetValue(stage, channel, conductance_nA)
    filename=conductanceDatabase.SaveIV(stage, channel, outBias, currents,conductance_nA)
    #send conductance info to the server for plotting    
    infos['datafolder']=filename
    infos['Stage']=stage 
    infos['electrode']=electrode
    conductanceDatabase.SendInfo(infos)
    return conductance_nA        

def Plate(infos,stage,channel,attempt,conductanceDatabase,plater):
    
    #reset everything so we can change the electrode without tripping the device
    plater.setBias(0)
    plater.setThreshold(current_nA = 200)
    plater.ResetDevice()
    plater.BottomElectrode()

    #running record of the current attempts accross the two sides
    currents=[]
    sideApplied =[]
    
    #keep track of the time to set a max time
    start = time.time()
    currentSide=0
    for i in range(infos['attemptsBeforeIV']):
        if infos['pulsed']:#this has not been tested yet.  Still working on the code
            conductance_nA_IV,_,curr=plater.runPulseTrain(maxVoltage_mV=infos['bias_V']*1000.0, pulseLengthS=.05,
                                                           totalLengthS=.1, checkVoltage_mV=100,
                                                           numberPulses=10, plot=True)

            print(conductance_nA_IV)
            tripped=conductance_nA_IV>infos['threshold_nA']
        else:
            #chose a different side to start for each attempt to try to keep things moving evenly accross the device
            if infos['bothElectrodes']:
                if (i+attempt)%2==0  :
                    plater.BottomElectrode()
                    currentSide=0
                else:
                    plater.TopElectrode()
                    currentSide=1

            #returns if timed out or tripped
            tripped, curr=plater.runConstantBias2(maxVoltage_mV=infos['bias_V']*1000.0,
                                                  threshold_nA=infos['threshold_nA'],
                                                  settleTime_S=infos['settleTime_S'],
                                                  maxTime_S=infos['maxTimeS'],
                                                  plot=True,
                                                  junctionName=infos['Device'])
            
            #check for a real trip, a timeout, or a shorted device
            tripped = tripped or (np.max(curr)<-.1) or (np.max(curr)>50)
            if (time.time()-start>infos['maxTime_Min']*60):
                break

        print('Iteration',i)
        if len(currents)==0 :
            currents=curr
            sideApplied=currentSide * np.ones(len(curr))
        else:
            currents=np.concatenate([currents,curr])
            sideApplied=np.concatenate([sideApplied,currentSide * np.ones(len(curr))])
            
        plt.plot( np.linspace(0, len(currents)/plater.samplesPerSec,len(currents)), currents)
        plt.show()

        if (time.time()-start>infos['maxTime_Min']*60):
            break
        
        if tripped :
            break

    
    conductanceDatabase.SavePlate(stage, channel, infos, attempt, currents, sideApplied, plater)
        
    return tripped,currents

class Plater:
    def __init__(self,samplesPerSecond):
        self.samplesPerSec=samplesPerSecond
        
        self.channelMap = {}
        self.channels=[]
        
        logicMap ={}
        
        for i in range(0,16):
            res = [int(i)==1 for i in list('{0:04b}'.format(i))]
            logicMap[i] =res[::-1]
            
        cc=0
        for chan in ['W','N']:
            for index in range(1,9):
                outs = [False]*2
                if cc>15:
                    outs[0]=True
                    outs=np.concatenate([outs,logicMap[cc-16]])
                else:
                    outs[1]=True
                    outs=np.concatenate([outs,logicMap[cc]])
                self.channelMap[chan+str(index)]=outs
                self.channels.append(chan+str(index))
                cc+=1
        for chan in ['E','S']:
            for index in range(1,9):
                outs = [False]*2
                if cc>15:
                    outs[0]=True
                    outs=np.concatenate([outs,logicMap[cc-16]])
                else:
                    outs[1]=True
                    outs=np.concatenate([outs,logicMap[cc]])
                self.channelMap[chan+str(index)]=outs
                self.channels.append(chan+str(index))
                cc+=1
        self.start()
        
    def startPlayRecord(self):
        self.iMonTask = nidaqmx.Task(new_task_name ='iMonTask')
        self.biasTask = nidaqmx.Task(new_task_name ='biasTask')
        self.iMonTask.ai_channels.add_ai_voltage_chan("Dev1/ai6")
        self.iMonTask.ai_channels.add_ai_voltage_chan("Dev1/_ao0_vs_aognd")
        self.iMonTask.timing.cfg_samp_clk_timing(self.samplesPerSec, source="", sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self.samplesPerSec)
        self.biasTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        
    def start(self):
        
        self.startPlayRecord()
        self.muxTask = nidaqmx.Task(new_task_name ='muxTask')
        self.topBottomTask = nidaqmx.Task(new_task_name ='topBottomTask')
        self.resetTask = nidaqmx.Task(new_task_name ='resetTask')
        self.controlTask = nidaqmx.Task(new_task_name ='controlTask')
        
        self.icutTask  = nidaqmx.Task(new_task_name ='icutTask')
        self.muxTask.do_channels.add_do_chan("Dev1/port0/line2:7", line_grouping=LineGrouping.CHAN_PER_LINE)
        self.topBottomTask.do_channels.add_do_chan("Dev1/port0/line1", line_grouping=LineGrouping.CHAN_PER_LINE)
        self.resetTask.do_channels.add_do_chan("Dev1/port0/line0", line_grouping=LineGrouping.CHAN_PER_LINE)
        self.icutTask.ao_channels.add_ao_voltage_chan("Dev1/ao1")
         
        self.setBias(0)
        self.setThreshold(10)
        
    def __enter__(self):
        return self 
 
    def __exit__(self, *args):
        print('Closing tasks')
        self.setBias(0)
        self.setThreshold(0)
        self.iMonTask.close()
        self.muxTask.close()
        self.topBottomTask.close()
        self.resetTask.close()
        self.controlTask.close()
        self.biasTask.close()
        self.icutTask.close()

    def ResetDevice(self ):
        self.resetTask.write([True ], auto_start=True)
        time.sleep(.1)
        self.resetTask.write([False ], auto_start=True)

    def TopElectrode(self ):
        self.topBottomTask.write([True ], auto_start=True)
        time.sleep(.1)

    def BottomElectrode(self ):
        self.topBottomTask.write([False ], auto_start=True)
        time.sleep(.1)    

    def SelectChannel(self,channel ):
        self.muxTask.write(self.channelMap[channel], auto_start=True)
        time.sleep(.1)

    def setBias(self,bias ):
        self.biasTask.write([bias], auto_start=True)

      

    def setThreshold(self,current_nA ):
        bias = current_nA/50.0
        print('Thresh Current:' + str(current_nA) + 'nA', 'bias:' + str(bias) + 'V')
        self.icutTask.write([bias], auto_start=True)
        

        
    def disableThreshold(self):
        self.resetTask.write([True ], auto_start=True)
        
    def runConstantBias(self,bias_V, threshold_nA, maxTimeS,delayTimeS=1,resetToZero=True,samplesPerPoint=200):
        output = np.zeros([ samplesPerPoint])
        currents=[]
        times =[]
        starts = time.time()
        slew_V_s=1000/1000
        secPerSample=1.0/self.samplesPerSec*samplesPerPoint
        segmentTime = bias_V/slew_V_s
        segmentPoints =int( segmentTime/secPerSample)
        biasi =   np.linspace(0,bias_V,segmentPoints) 
        self.iMonTask.start()
        maxA=0
        self.setThreshold(current_nA = 50)
        for bias in biasi:
            self.setBias(bias)
            self.iMonReader.read_many_sample(data = output,number_of_samples_per_channel = samplesPerPoint)# read from DAQ
            maxA =np.max(output)*50
            times.append( time.time()- starts)
            currents.append(maxA)
            
        self.setBias(bias_V)
        startDelay = time.time()
        while (time.time()- startDelay)<delayTimeS:
            self.iMonReader.read_many_sample(data = output,number_of_samples_per_channel = samplesPerPoint)# read from DAQ
            maxA =np.max(output)*50
            times.append( time.time()- starts)
            currents.append(maxA)
            
        self.setThreshold(current_nA = threshold_nA)
        while (time.time()- starts)<maxTimeS and maxA<threshold_nA:
            self.iMonReader.read_many_sample(data = output,number_of_samples_per_channel = samplesPerPoint)# read from DAQ
            maxA =np.max(output)*50
            times.append( time.time()- starts)
            currents.append(maxA)
            
        self.iMonTask.stop()
        if resetToZero:
            self.setBias(0)
        currents=np.array(currents)
        clear_output(wait=True)
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(times,currents, color=color)
        ax1.set_ylabel('Current (nA)', color=color)
        ax1.set_xlabel('Time (S)')
        fig.tight_layout() 
        plt.show()
        return maxA>threshold_nA,times,currents

    def runRT(self, voltage_mV, time_s):
        maxVoltage=voltage_mV/1000.0
        
        segmentPoints =int( time_s*self.samplesPerSec)
        biasi = np.zeros(segmentPoints)+maxVoltage
        
        outBias,currents=self.RunPlayRecord( biasi,time_s)
        
        conductance_nA= np.mean ( currents)/maxVoltage 
        
        return conductance_nA,  currents
    
    def RunPlayRecord(self, biasi,measureTime_s):
        totalPoints = int( self.samplesPerSec*measureTime_s )
        
        for task in (self.iMonTask, self.biasTask):
            task.timing.cfg_samp_clk_timing(
                rate=self.samplesPerSec, source="OnboardClock", samps_per_chan=totalPoints
            )

        # trigger write_task as soon as read_task starts
        self.biasTask.triggers.start_trigger.cfg_dig_edge_start_trig(
            self.iMonTask.triggers.start_trigger.term
        )
        
        # squeeze as Task.write expects 1d array for 1 channel
        self.biasTask.write(biasi , auto_start=False)
        # write_task doesn't start at read_task's start_trigger without this
        self.biasTask.start()
        # do not time out for long inputs
        currents = self.iMonTask.read(totalPoints, timeout=WAIT_INFINITELY)
        currents=np.asarray(currents)
        
        self.iMonTask.stop()
        self.biasTask.close()
        self.iMonTask.close()
        
        self.biasTask=None
        self.iMonTask=None
       
        
        self.startPlayRecord()
        return currents[1],currents[0]*50

    def runCV(self, startVoltage_V, minVoltage_V, maxVoltage_V, slew_mV_s ,cellOffset_V=2.5, plot=False):
        slew_V_s=slew_mV_s/1000.0
         
        
        segmentTime1 = np.abs(maxVoltage_V-startVoltage_V)/slew_V_s
        segmentPoints =int( segmentTime1*self.samplesPerSec)
        S1=np.linspace(startVoltage_V+cellOffset_V,maxVoltage_V+cellOffset_V,segmentPoints)

        segmentTime2 = np.abs(maxVoltage_V-minVoltage_V)/slew_V_s
        segmentPoints =int( segmentTime2*self.samplesPerSec)
        S2=np.linspace(maxVoltage_V+cellOffset_V, minVoltage_V+cellOffset_V,segmentPoints)

        segmentTime3 = np.abs(startVoltage_V-minVoltage_V)/slew_V_s
        segmentPoints =int( segmentTime3*self.samplesPerSec)
        S3=np.linspace(minVoltage_V+cellOffset_V, startVoltage_V+cellOffset_V,segmentPoints)

        
        biasi = np.concatenate( [S1,S2,S3])
         
        
        outBias,currents=self.RunPlayRecord( biasi,segmentTime1+segmentTime2+segmentTime3)
        
        fBias, fCurrents =  outBias[:int(len(currents)/2)],currents[:int(len(currents)/2)]
        fBias, fCurrents =fBias[fCurrents<180],fCurrents[fCurrents<180]
        fBias, fCurrents =fBias[int(len(fBias)/2):],fCurrents[int(len(fBias)/2):]
        conductance_nA=np.polyfit(fBias,fCurrents,1)[0] 
        if plot:
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(outBias-cellOffset_V,currents, color=color)
            ax1.set_ylabel('Current (nA)', color=color)
            ax1.set_xlabel('Voltage (V)')

            fig.tight_layout() 
            plt.show()

           
            print(str(conductance_nA), 'nS')
        
        return conductance_nA, outBias-cellOffset_V,currents     
    
    def runIV2(self, maxVoltage_mV, slew_mV_s , plot=False):
        slew_V_s=slew_mV_s/1000.0
        maxVoltage=maxVoltage_mV/1000.0
        
        segmentTime = maxVoltage/slew_V_s
       
        segmentPoints =int( segmentTime*self.samplesPerSec)
        biasi = np.concatenate( [np.linspace(0,maxVoltage,segmentPoints),np.linspace(maxVoltage,0,segmentPoints)])
         
        
        outBias,currents=self.RunPlayRecord( biasi,2*segmentTime)
        
        fBias, fCurrents =  outBias[:int(len(currents)/2)],currents[:int(len(currents)/2)]
        fBias, fCurrents =fBias[fCurrents<180],fCurrents[fCurrents<180]
        fBias, fCurrents =fBias[int(len(fBias)/2):],fCurrents[int(len(fBias)/2):]
        conductance_nA=np.polyfit(fBias,fCurrents,1)[0] 
        if plot:
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(outBias,currents, color=color)
            ax1.set_ylabel('Current (nA)', color=color)
            ax1.set_xlabel('Voltage (V)')

            fig.tight_layout() 
            plt.show()

           
            print(str(conductance_nA), 'nS')
        
        return conductance_nA, outBias,currents    
    
   
    
    def pulseBias(self,maxVoltage_mV, pulseLengthS,totalLengthS,numberPulses=1, plot=False ):
        maxVoltage=maxVoltage_mV/1000.0
        
        segmentPoints =int( totalLengthS*self.samplesPerSec)
        delay = int( self.samplesPerSec* (totalLengthS-pulseLengthS)/2.0)
        pulseSamples =int( self.samplesPerSec* pulseLengthS)
        
        biasi=[]
        for i in range(numberPulses):
            bias = np.zeros(segmentPoints)
            bias[delay:(delay+pulseSamples)]=maxVoltage
            biasi=np.concatenate([biasi,bias])
        
        outBias,currents=self.RunPlayRecord( biasi,totalLengthS*numberPulses)
        
        conductance_nA= np.mean ( currents [outBias>(maxVoltage/2.0)])/maxVoltage 
        
        if plot:
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('Bias (V)', color=color)
            ax1.plot(outBias, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()  

            color = 'tab:blue'
            ax2.set_ylabel('Current (nA)', color=color) 
            ax2.plot(currents, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            print(str(conductance_nA), 'nS')
        return conductance_nA, outBias,currents    
        
        
    def runPulseTrain(self,maxVoltage_mV, pulseLengthS,totalLengthS, checkVoltage_mV, numberPulses=1, plot=False):
        
        self.setThreshold(current_nA = 50)
        self.BottomElectrode()
        conductance_nA_B, _,currents_B=self.pulseBias( maxVoltage_mV=maxVoltage_mV, pulseLengthS = pulseLengthS,
                                                      totalLengthS=totalLengthS,numberPulses=numberPulses)
        
        self.ResetDevice()
        self.TopElectrode()
        conductance_nA_T, _,currents_T=self.pulseBias( maxVoltage_mV=maxVoltage_mV, pulseLengthS = pulseLengthS,
                                                      totalLengthS=totalLengthS,numberPulses=numberPulses)
        
        conductance_nA_IV, _,currents_IV=self.runIV2( maxVoltage_mV=checkVoltage_mV, slew_mV_s=200,plot=plot)
        
        currents = np.concatenate([currents_B,currents_T,currents_IV])
        
        if plot:
            plt.plot( np.linspace(0,len(currents)/self.samplesPerSec ,len(currents)), currents)
            plt.ylabel('Current (nA)')
            plt.xlabel('Time (s)')
            plt.show()
        
        return conductance_nA_IV,[conductance_nA_B,conductance_nA_T],currents
    
    def runRamp(self, maxVoltage_mV,slew_mV_s):
        slew_V_s=slew_mV_s/1000.0
        maxVoltage=maxVoltage_mV/1000.0
       
        segmentTime = maxVoltage/slew_V_s
        segmentPoints =int( segmentTime*self.samplesPerSec)
        biasi = np.concatenate( [np.linspace(0,maxVoltage,segmentPoints)])
        
        outBias_R,currents_R=self.RunPlayRecord( biasi,segmentTime)
        return outBias_R,currents_R
        
        
    def runConstantBias2(self,maxVoltage_mV, threshold_nA, maxTime_S,
                         settleTime_S=.5,
                         slew_mV_s=1000,resetToZero=True, plot=False, junctionName=''):
       
        currents =[]
        starts = time.time()
        #first turn the threshold down to allow the ramp to get voltage up to the value
        self.setThreshold(current_nA = 50)
        if slew_mV_s>0:
            outBias_R,currents_R = self.runRamp(maxVoltage_mV, slew_mV_s)
            currents.append(currents_R)
        else:
            self.setBias(maxVoltage_mV/1000.0)
            
         
        
        if settleTime_S>0:
            conductance_nA, current_S = self.runRT(voltage_mV=maxVoltage_mV, time_s=settleTime_S)
            currents.append(current_S)
        
        #turn down the threshold to do the deposition
        self.setThreshold(current_nA = threshold_nA)
        max_nA=0
        while (time.time()- starts)<maxTime_S and max_nA<threshold_nA:
            conductance_nA, current = self.runRT(voltage_mV=maxVoltage_mV, time_s=1)
            max_nA =np.max(current) 
            currents.append(current)
            
        currents=np.concatenate(currents)
        
        if plot:
            clear_output()
            plt.title(junctionName)
            plt.plot( np.linspace(0,len(currents)/self.samplesPerSec, len(currents)), currents, label='Measure')
            if slew_mV_s>0:
                x=np.linspace(0,len(currents_R)/self.samplesPerSec, len(currents_R))
                plt.plot(x , currents_R,label='Ramp')
            else:
                x=[0]
                
            if settleTime_S>0:
                x=np.linspace(0,len(current_S)/self.samplesPerSec, len(current_S))+x[-1]
                plt.plot(x, current_S,label='Settle')
                
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Current (nA)')
            plt.show()
        
        if resetToZero:
            self.setBias(0)
            
        
        return max_nA>threshold_nA, currents    