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

tickers = ['AAXJ', 'ACWI', 'AGG', 'AMLP', 'AOA', 'AOK', 'AOR', 'BIL', 'BKLN', 'BLV', 'BND', 'BSV', 'CORN', 'CQQQ',
           'DBA', 'DBO', 'DIA', 'DJP', 'DUST', 'DVY', 'EEM', 'EFA', 'EMB', 'ERX', 'EWG', 'EWH', 'EWJ', 'EWL', 'EWN',
           'EWT', 'EWW', 'EWY', 'EWZ', 'EZU', 'FAS', 'FAZ', 'FEZ', 'FXI', 'GDX', 'GDXJ', 'GLD', 'HDV', 'HYG', 'IAEX.L',
           'IAU','ABB', 'IDEM.L', 'IEF', 'IEML.L', 'IFFF.L', 'IHI', 'IJH', 'IJJ', 'IJPE.L', 'IJR', 'ILTB', 'IMEU.L',
           'IMIB.L', 'ISF.L', 'ITOT','ITWN.L', 'IUKP.L', 'IUSA.L', 'IUSG', 'IUSV', 'IVV', 'IVW', 'IWB', 'IWF', 'IWM',
           'IWN', 'IWO', 'IWR', 'IWS', 'IXJ', 'JNK', 'KBE','KRE', 'LIT', 'LQD', 'MCHI', 'MDY', 'MINT', 'MUB', 'NUGT',
           'OIH', 'PALL', 'PFF', 'PGX', 'PHYS', 'PPLT', 'PSLV', 'QQQ', 'RSX', 'RWR', 'SCHE', 'SCHF', 'SCHX', 'SCO',
           'SDIV', 'SDOW', 'SDS', 'SEMB.L', 'SH', 'SHV', 'SHY', 'SKYY', 'SLV', 'SMH', 'SOXL', 'SOXS', 'SOXX', 'SPLV',
           'SPXL', 'SPXS', 'SPXU', 'SPY', 'SPYG', 'SQQQ', 'SRTY', 'SSO', 'SWDA.L', 'TAN', 'TFI', 'THD', 'TIP', 'TLT',
           'TMF', 'TNA', 'TQQQ', 'TVIX', 'TZA', 'UCO', 'UDOW', 'UGA', 'UNG', 'UPRO', 'USL', 'USO', 'USRT', 'VB', 'VBK',
           'VBR', 'VCIT', 'VCSH', 'VEA', 'VEU', 'VFH', 'VGK', 'VGT', 'VHT', 'VIG', 'VNQ', 'VO', 'VOE', 'VONG', 'VOO',
           'VOOG', 'VOOV', 'VOX', 'VTI', 'VTV', 'VUG', 'VWO', 'VXUS', 'XBI', 'XCX5.L', 'XLB', 'XLE', 'XLF', 'XLI',
           'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XOP', 'XS6R.L', 'YINN']