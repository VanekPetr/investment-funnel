#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:59:42 2020

@author: Petr Vanek
"""

import pandas as pd
import numpy as np


# Function computing the geometric mean of annual returns
def meanRetAn(data):             
    Result = 1
    
    for i in range(len(data.index)):
        Result *= (1+data.iloc[i,:])
        
    Result = Result**(1/float(len(data.index)/52))-1
     
    return Result

# Function computing the final statistics of the backtesting
def finalStat(data):
    # TABLE WITH AVG RET AND STD OF RET
    data = data.pct_change()                  
    data = data.drop(data.index[:1])
    
    mu_ga = round(meanRetAn(data), 2)                     # annual geometric mean
    stdev_a = round(data.std(axis=0) * np.sqrt(52), 2)    # standard deviation of Annual Returns

    statDf = pd.concat([mu_ga,stdev_a], axis=1)         # table
    statName = ["Average Annual Returns","Standard Deviation of Returns"]
    statDf.columns = statName                           # add names
    
    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = round(statDf.loc[:,"Average Annual Returns"]/statDf.loc[:,"Standard Deviation of Returns"], 2)
    statDf = pd.concat([statDf,sharpe], axis=1)         # add sharpe ratio into the table
    statName = ["Avg An Ret", "Std Dev of Ret", "Sharpe R"]
    statDf.columns = statName
    
    return statDf

# Function returning weekly returns
def getWeeklyRet(data):
    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    workPrices = data
    # MODIFY THE DATA
    pricesWed = workPrices[workPrices.index.weekday==2]             # Only wednesdays

    # Get weekly returns
    weeklyReturns = pricesWed.pct_change()                  
    weeklyReturns = weeklyReturns.drop(weeklyReturns.index[:1])     # drop first NaN row

    return weeklyReturns

# TICKERS OF ETFs WE ARE GOING TO WORK WITH
tickers = ["ANGL","ASHR","BIV","BKLN","BNDX","BOND","BRF","CGW","CMBS",
           "CMF","CORP","CSM","CWB","DBA","DBB","DBO","DBS","DBV","DES","DGL"]
    # ,
    #        "DGRW","DIA","DLS","DOL","DON","DSI","DXGE","DXJ","EBND","ECH","EDEN",
    #        "EEM","EFV","EIDO","EIRL","ENZL","EPHE","EWD","EWG","EWH","EWI","EWL"]
           # "EWM","EWN","EWQ","EWS","EWT","EWU","EWW","EWY","EZA","FBT","FCG",
           # "FCOM","FDD","FDL","FEU","FEX","FLRN","FPX","FTA","FTCS","FTSM","FXA",
           # "FXB","FXC","FXE","FXF","FXR","FXY","FXZ","GBF","GNMA","GREK","GSY",
           # "HDV","HEDJ","HEFA","HYD","HYEM","IAU","IBND","IDLV","IGE","IGN","IGV",
           # "IHI","INDA","IOO","IPE","IPFF","IQDF","ISTB","ITA","ITB","ITM","IUSV",
           # "IVOO","IVOV","IWC","IWN","IWO","IWY","IXG","IXN","IYE","IYY","IYZ",
           # "JKD","JKE","JKG","JPXN","KBWP","KOL","KRE","KXI","LTPZ","MCHI","MDYG",
           # "MDYV","MGV","MLPA","MOAT","MOO","MTUM","OIH","PALL","PCY","PDP","PEY",
           # "PFXF","PHB","PHO","PJP","PKW","PRFZ","PSCC","PSCT","PSCU","PSK","PSL",
           # "PUI","PWB","PWV","PWZ","QDF","QUAL","RDIV","REM","REZ","RFG","RING",
           # "RSX","RTH","RWJ","RWL","RWX","RXI","RYF","SCHC","SCHE","SCJ","SDIV",
           # "SDOG","SGDM","SGOL","SHM","SILJ","SIZE","SLQD","SLY","SLYG","SMLV",
           # "SNLN","SOCL","SPHQ","SPYG","TAN","TDIV","TDTT","THD","TIP","TOK","TUR",
           # "UGA","URA","URTH","USDU","VBK","VCLT","VEA","VLUE","VNM","VOE","VONE",
           # "VONG","VONV","VOT","VXF","XBI","XES","XHS","XLE","XLG","XLI","XLK",
           # "XLP","XLU","XLV","XLY","XME","XPH","XRT","XSD","XTN","ZROZ"]
