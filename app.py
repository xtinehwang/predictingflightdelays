
# coding: utf-8

# In[29]:


import os
import numpy as np
import flask
import pickle
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.datastructures import ImmutableMultiDict
import datetime 
import calendar 
from sklearn.preprocessing import StandardScaler

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/result')
def result():
    imd = flask.request.args
    x = imd.to_dict(flat=False)
    
    with open("model.pkl", "rb") as f:
        logmodel = pickle.load(f)
    
    prcpDF = pd.read_csv('prcp.csv')
    prcpDF = prcpDF.set_index('Unnamed: 0')
    grimeDF = pd.read_csv('glrime.csv')
    grimeDF = grimeDF.set_index('Unnamed: 0')
    busAir = pd.read_csv('busAir.csv')
    busAir =  busAir.set_index('airport')
    
    featuresDict = dict.fromkeys(['O_PRCP', 'O_glazeRime', 'PRCP', 'glazeRime', 'ATL_o', 'AUS_o',
       'BNA_o', 'BOS_o', 'BWI_o', 'CLE_o', 'CLT_o', 'CMH_o', 'CVG_o',
       'DAL_o', 'DCA_o', 'DEN_o', 'DFW_o', 'DTW_o', 'EWR_o', 'FLL_o',
       'HNL_o', 'HOU_o', 'IAD_o', 'IAH_o', 'IND_o', 'JFK_o', 'LAS_o',
       'LAX_o', 'LGA_o', 'MCI_o', 'MCO_o', 'MDW_o', 'MIA_o', 'MSP_o',
       'MSY_o', 'OAK_o', 'ORD_o', 'PDX_o', 'PHL_o', 'PHX_o', 'PIT_o',
       'RDU_o', 'RSW_o', 'SAN_o', 'SAT_o', 'SEA_o', 'SFO_o', 'SJC_o',
       'SJU_o', 'SLC_o', 'SMF_o', 'SNA_o', 'STL_o', 'TPA_o', 'ATL', 'AUS',
       'BNA', 'BOS', 'BWI', 'CLE', 'CLT', 'CMH', 'CVG', 'DAL', 'DCA',
       'DEN', 'DFW', 'DTW', 'EWR', 'FLL', 'HNL', 'HOU', 'IAD', 'IAH',
       'IND', 'JFK', 'LAS', 'LAX', 'LGA', 'MCI', 'MCO', 'MDW', 'MIA',
       'MSP', 'MSY', 'OAK', 'ORD', 'PDX', 'PHL', 'PHX', 'PIT', 'RDU',
       'RSW', 'SAN', 'SAT', 'SEA', 'SFO', 'SJC', 'SJU', 'SLC', 'SMF',
       'SNA', 'STL', 'TPA', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
       'Sun', '0_h', '1_h', '2_h', '3_h', '4_h', '5_h', '6_h', '7_h',
       '8_h', '9_h', '10_h', '11_h', '12_h', '13_h', '14_h', '15_h',
       '16_h', '17_h', '18_h', '19_h', '20_h', '21_h', '22_h', '23_h',
       'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
       'Oct', 'Nov', 'Dec', 'AA', 'AS', 'B6', 'DL', 'F9', 'G4', 'HA',
       'NK', 'UA', 'VX', 'WN', 'board', 'EWRweight', 'SFOweight',
       'LGAweight', 'PRCPnew', 'GRP', 'logboard', 'ORDPRCP'])
    
    featuresDict = featuresDict.fromkeys(featuresDict, 0)
    
    def findDay(date): 
        born = datetime.datetime.strptime(date, '%Y-%m-%d').weekday() 
        return (calendar.day_name[born]) 
    
    airlineCat = str(x['airline']).replace("[",'').replace("]",'').replace("'",'')
    arrivalAirport = str(x['d_airport']).replace("[",'').replace("]",'').replace("'",'')
    originAirport = str(x['o_airport']).replace("[",'').replace("]",'').replace("'",'')+'_o'
    originAirportMinusO = str(x['o_airport']).replace("[",'').replace("]",'').replace("'",'')
    month = datetime.date(1900, int((str(x['date']).replace("[",'').replace("]",'').replace("'",'').replace('-',''))[4:6]), 1).strftime('%B')[0:3]
    dOW = findDay(str(x['date']).replace("[",'').replace("]",'').replace("'",''))
    hourOfFlight = str(x['time']).replace("[",'').replace("]",'').replace("'",'').replace('-','')[0:2]+'_h'
    prcpValueO = prcpDF.loc[originAirportMinusO][month]
    prcpValueA = prcpDF.loc[arrivalAirport][month]
    grimeO = grimeDF.loc[originAirportMinusO][month]
    grimeA = grimeDF.loc[arrivalAirport][month]
    boardNum = int(busAir.loc[originAirportMinusO]['boarding'].replace(",",""))
    
    featuresDict[airlineCat] = 1
    featuresDict[arrivalAirport] = 1
    featuresDict[originAirport] = 1
    featuresDict[month] =  1
    featuresDict[dOW] =  1
    featuresDict[hourOfFlight] = 1
    featuresDict['PRCP'] = prcpValueA
    featuresDict['O_PRCP'] = prcpValueO
    featuresDict['glazeRime'] = grimeA
    featuresDict['O_glazeRime'] = grimeO
    featuresDict['board'] = boardNum

    featuresDict['EWRweight'] = featuresDict['EWR_o'] * 10
    featuresDict['SFOweight'] = featuresDict['SFO_o'] * 5
    featuresDict['LGAweight'] = featuresDict['LGA_o'] * 3
    featuresDict['PRCPnew'] = featuresDict['PRCP'] * featuresDict['O_PRCP']* 100
    featuresDict['GRP'] = featuresDict['O_glazeRime'] ** featuresDict['O_PRCP']
    featuresDict['logboard'] = np.sqrt(featuresDict['board'])
    
    delayPrediction = [[featuresDict['O_PRCP'], featuresDict['O_glazeRime'], featuresDict['PRCP'],
                   featuresDict['glazeRime'], featuresDict['ATL_o'], featuresDict['AUS_o'],
                   featuresDict['BNA_o'], featuresDict['BOS_o'], featuresDict['BWI_o'], 
                   featuresDict['CLE_o'], featuresDict['CLT_o'], featuresDict['CMH_o'], 
                   featuresDict['CVG_o'], featuresDict['DAL_o'], featuresDict['DCA_o'], 
                   featuresDict['DEN_o'], featuresDict['DFW_o'], featuresDict['DTW_o'], featuresDict['EWR_o'], 
                   featuresDict['FLL_o'], featuresDict['HNL_o'], featuresDict['HOU_o'], featuresDict['IAD_o'], 
                   featuresDict['IAH_o'], featuresDict['IND_o'], featuresDict['JFK_o'], featuresDict['LAS_o'],
                   featuresDict['LAX_o'], featuresDict['LGA_o'], featuresDict['MCI_o'], featuresDict['MCO_o'], 
                   featuresDict['MDW_o'], featuresDict['MIA_o'], featuresDict['MSP_o'], featuresDict['MSY_o'], 
                   featuresDict['OAK_o'], featuresDict['ORD_o'], featuresDict['PDX_o'], featuresDict['PHL_o'], 
                   featuresDict['PHX_o'], featuresDict['PIT_o'], featuresDict['RDU_o'], featuresDict['RSW_o'], 
                   featuresDict['SAN_o'], featuresDict['SAT_o'], featuresDict['SEA_o'], featuresDict['SFO_o'], 
                   featuresDict['SJC_o'], featuresDict['SJU_o'], featuresDict['SLC_o'], featuresDict['SMF_o'], 
                   featuresDict['SNA_o'], featuresDict['STL_o'], featuresDict['TPA_o'], featuresDict['ATL'], 
                   featuresDict['AUS'], featuresDict['BNA'], featuresDict['BOS'], featuresDict['BWI'], 
                   featuresDict['CLE'], featuresDict['CLT'], featuresDict['CMH'], featuresDict['CVG'], 
                   featuresDict['DAL'], featuresDict['DCA'], featuresDict['DEN'], featuresDict['DFW'], 
                   featuresDict['DTW'], featuresDict['EWR'], featuresDict['FLL'], featuresDict['HNL'], 
                   featuresDict['HOU'], featuresDict['IAD'], featuresDict['IAH'], featuresDict['IND'], 
                   featuresDict['JFK'], featuresDict['LAS'], featuresDict['LAX'], featuresDict['LGA'], 
                   featuresDict['MCI'], featuresDict['MCO'], featuresDict['MDW'], featuresDict['MIA'],
                   featuresDict['MSP'], featuresDict['MSY'], featuresDict['OAK'], featuresDict['ORD'], 
                   featuresDict['PDX'], featuresDict['PHL'], featuresDict['PHX'], featuresDict['PIT'], 
                   featuresDict['RDU'], featuresDict['RSW'], featuresDict['SAN'], featuresDict['SAT'], 
                   featuresDict['SEA'], featuresDict['SFO'], featuresDict['SJC'], featuresDict['SJU'], 
                   featuresDict['SLC'], featuresDict['SMF'], featuresDict['SNA'], featuresDict['STL'], 
                   featuresDict['TPA'], featuresDict['Monday'], featuresDict['Tuesday'], featuresDict['Wednesday'], 
                   featuresDict['Thursday'], featuresDict['Friday'], featuresDict['Saturday'],featuresDict['Sun'], 
                   featuresDict['0_h'], featuresDict['1_h'], featuresDict['2_h'], featuresDict['3_h'], featuresDict['4_h'], 
                   featuresDict['5_h'], featuresDict['6_h'], featuresDict['7_h'],featuresDict['8_h'], featuresDict['9_h'], 
                   featuresDict['10_h'], featuresDict['11_h'], featuresDict['12_h'], featuresDict['13_h'], 
                   featuresDict['14_h'], featuresDict['15_h'],featuresDict['16_h'], featuresDict['17_h'], featuresDict['18_h'],
                   featuresDict['19_h'], featuresDict['20_h'], featuresDict['21_h'], featuresDict['22_h'], 
                   featuresDict['23_h'],featuresDict['Jan'], featuresDict['Feb'], featuresDict['Mar'], featuresDict['Apr'], 
                   featuresDict['May'], featuresDict['Jun'], featuresDict['Jul'], featuresDict['Aug'], featuresDict['Sep'],
                   featuresDict['Oct'], featuresDict['Nov'], featuresDict['Dec'], featuresDict['AA'], featuresDict['AS'], 
                   featuresDict['B6'], featuresDict['DL'], featuresDict['F9'], featuresDict['G4'], featuresDict['HA'],
                   featuresDict['NK'], featuresDict['UA'], featuresDict['VX'], featuresDict['WN'], featuresDict['board'], 
                   featuresDict['EWRweight'], featuresDict['SFOweight'], featuresDict['LGAweight'], 
                   featuresDict['PRCPnew'], featuresDict['GRP'], featuresDict['logboard'], 
                                         featuresDict['ORDPRCP']]]
    
    with open("scalefit.pkl", "rb") as f:
        scFIT = pickle.load(f)

    fittedData = scFIT.transform(np.asarray(delayPrediction))
    
    prediction = logmodel.predict(fittedData)
    
    if int(prediction) == 1:
        prediction = 'DELAYED'
    else:
        prediction = 'ON-TIME'

    print(prediction) 
        
    return flask.render_template('result.html',prediction=prediction, originAirportMinusO=originAirportMinusO,arrivalAirport=arrivalAirport)
    

app.run(threaded=True, port=5000)

