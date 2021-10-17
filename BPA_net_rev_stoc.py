# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:29:19 2019

@author: sdenaro
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
import numpy.matlib as matlib
import seaborn as sns
from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

#Set Preference Customers reduction percent (number)
custom_redux=0

# Yearly firm loads (aMW)
# upload BPA firm load column from file
df_load=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=0,skiprows=[0,1], usecols=[9])
#Save as Preference Firm (PF), Industrial Firm (IF) an Export (ET)
PF_load_y=df_load.loc[[13]].values - custom_redux*df_load.loc[[13]].values
IP_load_y=df_load.loc[[3]].values - custom_redux* df_load.loc[[3]].values
ET_load_y=df_load.loc[[14]]

# Hourly hydro generation from FCRPS stochastic simulation
#df_hydro=pd.read_csv('../../CAPOW/CAPOW_SD/Stochastic_engine/PNW_hydro/FCRPS/BPA_owned_dams.csv', header=None)
df_hydro=pd.read_csv('new_BPA_hydro_daily.csv', usecols=([1]))
BPA_hydro=pd.DataFrame(data=df_hydro.loc[0:365*1200-1,:].sum(axis=1)/24, columns=['hydro'])
BPA_hydro[BPA_hydro>45000]=45000
#Remove CAISO bad_years
BPA_hydro=pd.DataFrame(np.reshape(BPA_hydro.values, (365,1200), order='F'))
BPA_hydro.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
#reshuffle 
#BPA_hydro[[1, 122, 364, 543]]=BPA_hydro[[16, 126, 368, 547]]
BPA_hydro=pd.DataFrame(np.reshape(BPA_hydro.values, (365*1188), order='F'))


# Yearly resources other than hydro (aMW)
df_resources=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=1,skiprows=[0,1], usecols=[9])
Nuc_y=df_resources.loc[[7]]
Wind_y=df_resources.loc[[8]]
Purch_y=df_resources.loc[[10]]

# Yearly costs and monthly rates (Oct-Sep)
costs_y=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=3,skiprows=[0,3,4,5], usecols=[8])*pow(10,3)
PF_rates=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=4,skiprows=np.arange(13,31), usecols=[0,7])
PF_rates.columns=['month','2018']
IP_rates=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=5,skiprows=np.arange(13,31), usecols=[0,7])
IP_rates.columns=['month','2018']



#load BPAT hourly demand and wind and convert to daily
df_synth_load=pd.read_csv('../../CAPOW/CAPOW_SD/Stochastic_engine/Synthetic_demand_pathflows/Sim_hourly_load.csv', usecols=[1])
BPAT_load=pd.DataFrame(np.reshape(df_synth_load.values, (24*365,1200), order='F'))
base = dt(2001, 1, 1)
arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
BPAT_load.index=arr
BPAT_load=BPAT_load.resample('D').mean()
BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
#reshuffle
#BPAT_load[[1, 122, 364, 543]]=BPAT_load[[16, 126, 368, 547]]
BPAT_load=pd.DataFrame(np.reshape(BPAT_load.values, (365*1188), order='F'))


df_synth_wind=pd.read_csv('../../CAPOW/CAPOW_SD/Stochastic_engine/Synthetic_wind_power/wind_power_sim.csv', usecols=[1])
BPAT_wind=pd.DataFrame(np.reshape(df_synth_wind.values, (24*365,1200), order='F'))
BPAT_wind.index=arr
BPAT_wind=BPAT_wind.resample('D').mean()
BPAT_wind.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
#BPAT_wind[[1, 122, 364, 543]]=BPAT_wind[[16, 126, 368, 547]]
BPAT_wind=pd.DataFrame(np.reshape(BPAT_wind.values, (365*1188), order='F'))

# Calculate daily BPAT proportions for demand and wind
load_ratio=BPAT_load/BPAT_load.mean()
wind_ratio=BPAT_wind/BPAT_wind.mean()

# Derive daily BPA loads and other resources
y=2018
PF_load=pd.DataFrame(PF_load_y*load_ratio)
PF_load_avg=(np.reshape(PF_load.values, (365,1188), order='F')).sum(axis=0).mean()
IP_load=pd.DataFrame(IP_load_y*load_ratio)
IP_load_avg=(np.reshape(IP_load.values, (365,1188), order='F')).sum(axis=0).mean()
ET_load=pd.DataFrame(ET_load_y.loc[14,y]*load_ratio)
Purch=pd.DataFrame(Purch_y.loc[10,y]*load_ratio)
Wind=pd.DataFrame(Wind_y.loc[8,y]*wind_ratio)
Nuc=pd.DataFrame(data=np.ones(len(Wind))*Nuc_y.loc[7,y], index=Wind.index)

# STOCHASTIC MIdC and California daily prices
#MidC=pd.read_csv('../../CAPOW/CAPOW_SD/UCED/LR/MidC_daily_prices.csv').iloc[:, 1:]
MidC=pd.read_csv('MidC_daily_prices_new.csv').iloc[:, 1]
MidC=pd.DataFrame(np.reshape(MidC.values, (365,1200), order='F'))
MidC.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
#reshuffle
#MidC[[1, 122, 364, 543]]=MidC[[16, 126, 368, 547]]
MidC=pd.DataFrame(np.reshape(MidC.values, (365*1188), order='F'))
CAISO=pd.read_csv('../../CAPOW/CAPOW_SD/UCED/LR/CAISO_daily_prices.csv').iloc[:, 1:]
#reshuffle
#CAISO[['1', '122', '364', '543']]=CAISO[['16', '126', '368', '547']]
CAISO=pd.DataFrame(np.reshape(CAISO.values, (365*1188), order='F'))
Wholesale_Mkt=pd.concat([MidC,CAISO], axis=1)
Wholesale_Mkt.columns=['MidC','CAISO']
                        
# Extra regional discount and Transmission Availability
ExR=0.71
TA=1000

##Calculate revenue
start_res=158.7*pow(10,6)  #initial reserves as of 2018
Treas_fac1=320*pow(10,6)   # Treasury facility (1)
Treas_fac2=430*pow(10,6)   # Treasury facility (2)
starting_BA = 2.421*pow(10,9)  #Remaining borrowing authority as of 2018
trans_BA= 9.782*pow(10,6)*0.4 #40 percent contribution to BA from transmission line
Used_TF= 0                #used TF over the 20 year enesemble
#total BPA resources
trans_losses=3*(Wind + BPA_hydro + Nuc)/100; #consider 3% transmission losses
BPA_res=pd.DataFrame(data=(Wind + BPA_hydro + Purch + Nuc)-trans_losses)
#Calculate Surplus/Deficit given BPA resources BP_res
SD =pd.DataFrame(data=BPA_res - (PF_load + IP_load + ET_load))

months=pd.date_range('2001-1-1','2001-12-31', freq='D').strftime('%B').tolist()
months= np.transpose(matlib.repmat(months,1,1188))
PF_load['month']=months
IP_load['month']=months

#initialize
BPA_rev_d=pd.DataFrame(index=PF_load.index)
BPA_Net_rev_y=pd.DataFrame(index=np.arange(1,1189))
PF_rev=pd.DataFrame(index=PF_load.index)
IP_rev=pd.DataFrame(index=IP_load.index)
P=pd.DataFrame(index=SD.index)
SS=pd.DataFrame(index=SD.index)
Reserves=pd.DataFrame(index=np.arange(1,1189))
Reserves.loc[1,0]=start_res
TF=pd.DataFrame(index=np.arange(1,1189), columns=['TF1','TF2'])
TF.loc[:,:]=0
TTP=pd.DataFrame(index=np.arange(1,1189), columns=['TTP'])
TTP.loc[:]=True
Remaining_BA =pd.DataFrame(index=np.arange(1,1189))
Remaining_BA.loc[1,0]=starting_BA 
CRAC=0
CRAC_y=pd.DataFrame(index=np.arange(1,1189))
CRAC_y.loc[:,0]=0
CRAC_rev=pd.DataFrame(index=np.arange(1,1189))
CRAC_rev.loc[:,0]=0

#Create DataFrame list to hold results
Result_list = ['ensemble' + str(e) for e in range(1,60,1)]
Result_ensembles_y = {} 
Result_ensembles_d = {} 


p=10  # percent surplus that goes to reserve
p2=32  # percent surplus that goes to debt opt
d=1
e=1

def calculate_CRAC(NR_, tot_load):
    if NR_ > 5*pow(10,6):  
        if NR_ > 100*pow(10,6):
            NR1=100*pow(10,6)
            NR2=(NR_ - 100*pow(10,6))/2
        else: 
            NR1 = NR_
            NR2= 0
        X=min((NR1+NR2)/(tot_load*24) ,  300*pow(10,6)/(tot_load*24))
    else:
        X=0
    return X
    
    
        
    
for i in SD.index:
        #daily simulation
        # Calculate revenue from Obligations
        RatePF = PF_rates[str(y)][PF_rates['month']==months[i,0]].values 
        RatePF += CRAC
        PF_rev.loc[i,0]=PF_load.loc[i,0]*RatePF*24
        RateIP = IP_rates[str(y)][IP_rates['month']==months[i,0]].values 
        RateIP += CRAC
        IP_rev.loc[i,0]=IP_load.loc[i,0]*RateIP*24
        # Calculate Surplus/Deficit revenue
        if SD.loc[i,0]<0:
            if Wholesale_Mkt.loc[i,'CAISO']>Wholesale_Mkt.loc[i,'MidC']:
                    P.loc[i,0]=SD.loc[i,0]*Wholesale_Mkt.loc[i,'MidC']*24
            else:
                    P.loc[i,0]=SD.loc[i,0]*Wholesale_Mkt.loc[i,'CAISO']*24
            SS.loc[i,0]=0
        else:
                P.loc[i,0]=0
                if Wholesale_Mkt.loc[i,'CAISO']>Wholesale_Mkt.loc[i,'MidC']:
                    Ex=min(SD.loc[i,0],TA)
                else:
                    Ex=0
                SS.loc[i,0]=ExR*(Ex* Wholesale_Mkt.loc[i,'CAISO']*24) + (SD.loc[i,0]-Ex)*Wholesale_Mkt.loc[i,'MidC']*24
        BPA_rev_d.loc[i,0]= PF_rev.loc[i,0] + IP_rev.loc[i,0] + SS.loc[i,0] + P.loc[i,0]
        
        #yearly simulation        
        if ((i+1)/365).is_integer():
            year=int((i+1)/365)
            print(str(year))
            bol=year%2 == 0 
            PF_load_i = PF_load.iloc[(year-1)*365:year*365,0].sum()
            IP_load_i = IP_load.iloc[(year-1)*365:year*365,0].sum()
            tot_load_i = PF_load_i + IP_load_i
            BPA_Net_rev_y.loc[year,0]=(BPA_rev_d.loc[i-364:i,0]).sum() - costs_y.values
            if int(BPA_Net_rev_y.loc[year,0]<0):
               losses=-BPA_Net_rev_y.loc[year,0]
               Net_res1= Reserves.loc[year,0] - losses
               Reserves.loc[year+1,0] = max(Net_res1, 0)
               if int(Net_res1 < 0):
                   losses=-Net_res1
                   if (Remaining_BA.loc[year,0] - Used_TF) > 750*pow(10,6): #if TF is viable
                       TF.loc[year+1,'TF1']=min(losses , Treas_fac1-TF.TF1[year]*bol)
                       Used_TF+=TF.loc[year+1,'TF1']
                       if (Treas_fac1-TF.TF1[year]*bol - losses)<0:
                           losses= - (Treas_fac1-TF.TF1[year]*bol - losses)
                           CRAC+=calculate_CRAC(losses, tot_load_i)
                           #set max crac as +5$/MWh per ensemble
                           CRAC=min(CRAC, 5)
                           TF.loc[year+1,'TF2']=min(losses , Treas_fac2-TF.TF2[year]*bol)
                           Used_TF+=TF.loc[year+1,'TF2']
                           if (Treas_fac2-TF.TF2[year]*bol - losses) <= 0:
                               TTP.loc[year]=False
                   else:
                       print('Warning: depleted borrowing authority and deferred TP')
                       CRAC+=calculate_CRAC(losses, tot_load_i)
                       #set max crac as +5$/MWh per ensemble
                       CRAC=min(CRAC, 5)
                       TTP.loc[year]=losses
                               
            else:
               Reserves.loc[year+1,0]= min(Reserves.loc[year,0] + 0.01*p*BPA_Net_rev_y.loc[year,0],608691000-Treas_fac1) 
               #Remaining_BA.loc[year+1,0]= Remaining_BA.loc[year,0] + 0.01*p2*BPA_Net_rev_y.loc[year,0]  #debt optimization
               print('Debt optimization, added: ' + str( 0.01*p2*BPA_Net_rev_y.loc[year,0]))
            CRAC_y.loc[year+1]=CRAC
            Remaining_BA.loc[year+1,0]= max(0, Remaining_BA.loc[year,0] - 484*pow(10,6) + trans_BA, Remaining_BA.loc[year,0] - 484*pow(10,6) + trans_BA + 0.01*p2*BPA_Net_rev_y.loc[year,0])
            
            #ensembles            
            if year%20 == 0:
                Result_ensembles_y['ensemble' + str(e)] = pd.DataFrame(data= np.stack([Reserves.loc[year-19:year,0],TTP.loc[year-19:year,'TTP'],Remaining_BA.loc[year-19:year,0], TF.loc[year-19:year,'TF1'],TF.loc[year-19:year,'TF2'],
                                                                        CRAC_y.loc[year-19:year,0],BPA_Net_rev_y.loc[year-19:year,0]], axis=1), 
                                                                        columns=['Reserves','TTP','BA','TF1','TF2','CRAC','Net_Rev'])
                Result_ensembles_y['ensemble' + str(e)].reset_index(inplace=True, drop=True)
                Result_ensembles_d['ensemble' + str(e)]=pd.DataFrame(np.stack([BPA_rev_d.iloc[(year-19)*365:year*365,0],PF_rev.iloc[(year-19)*365:year*365,0], 
                                                                       IP_rev.iloc[(year-19)*365:year*365,0] ,  P.iloc[(year-19)*365:year*365,0], SS.iloc[(year-19)*365:year*365,0] ],axis=1),
                                                                       columns=['Rev_gross','PF_rev','IP_rev','P','SS'])
                Result_ensembles_d['ensemble' + str(e)].reset_index(inplace=True, drop=True)
                #initialize new ensemble
                e+=1
                Reserves.loc[year+1,0]=start_res
                CRAC=0 
                CRAC_y.loc[year+1]=0
                Used_TF=0    #initialize treasury facility
                Remaining_BA.loc[year+1,0] = starting_BA  #Initialize Remaining borrowing authority 



#Save results
Results_d=pd.DataFrame(np.stack([BPA_rev_d[0],PF_rev[0],IP_rev[0],P[0],SS[0],BPA_hydro[0],PF_load[0],IP_load[0],SD[0],BPA_res[0],Wholesale_Mkt['MidC'],Wholesale_Mkt['CAISO']],axis=1),
                       columns=['Rev_gross','PF_rev','IP_rev','P','SS','BPA_hydro','PF_load','IP_load','Surplus/Deficit','BPA_resources','MidC','CAISO' ])

with pd.ExcelWriter('BPA_net_rev_stoc_d_NEW.xlsx' ) as writer:
    Results_d.to_excel(writer, sheet_name='Results_d')
    for e in range (1,60):
        Result_ensembles_d['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))

with pd.ExcelWriter('BPA_net_rev_stoc_y_NEW.xlsx' ) as writer:
    for e in range (1,60):
        Result_ensembles_y['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
    costs_y.to_excel(writer,sheet_name='Costs_y')




