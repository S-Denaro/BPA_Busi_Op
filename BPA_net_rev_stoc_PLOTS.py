# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:41:09 2019

@author: sdenaro
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
from numpy import matlib as matlib
import seaborn as sns
import statsmodels.api as sm
sns.set(style='whitegrid')
import matplotlib.cm as cm 
#from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2  


#Set Preference Customers reduction percent ('' or '_minus10' or '_minus20')
redux='_NEW'
##Load results
Results_d= pd.read_excel('BPA_net_rev_stoc_d' + redux + '.xlsx', sheet_name='Results_d')
#for e in range (1,60):
#    Result_ensembles_d['ensemble' + str(e)]=pd.read_excel(BPA_net_rev_stoc_d' + redux + '.xlsx', sheet_name='ensemble' + str(e))
#    print(str(e))
#
#for e in range (1,60):
#    Result_ensembles_y['ensemble' + str(e)]=pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e))
#    print(str(e))
#    
#costs_y=pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx',sheet_name='Costs_y')
#PF_rates_avg=35.460833 
#IP_rates_avg=44.030833


#Results Yearly Aggregates
Calendar_year=np.reshape(matlib.repmat(np.arange(1,1189),365,1), 1188*365, 'C' )    
#PF_rev_y=Results_d.PF_rev.groupby(Calendar_year).sum()
#IP_rev_y=Results_d.IP_rev.groupby(Calendar_year).sum()
#SS_y=Results_d.SS.groupby(Calendar_year).sum()
#P_y=Results_d.P.groupby(Calendar_year).sum()
#BPA_hydro_y=Results_d.BPA_hydro.groupby(Calendar_year).sum()
PF_load_y=Results_d.PF_load.groupby(Calendar_year).sum()
IP_load_y=Results_d.IP_load.groupby(Calendar_year).sum()
MidC_y=Results_d.MidC.groupby(Calendar_year).mean()
CAISO_y=Results_d.CAISO.groupby(Calendar_year).mean()


Net_rev=pd.DataFrame(columns=['Net_Rev'])
for e in range (1,60):
    Net_rev=Net_rev.append(pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7]))
Net_rev.reset_index(inplace=True, drop=True)
Net_rev['positive']=Net_rev['Net_Rev']>0
Net_rev['negative']=Net_rev['Net_Rev']<0


####Weather data
#df_weather=pd.read_csv('../../CAPOW/CAPOW_SD/Stochastic_engine/Synthetic_weather/INDEX_synthetic_temp_wind.csv')
#df_weather.index=pd.DatetimeIndex(data=(t for t in dates if not isleap(t.year)))
#df_weather=df_weather[BPA_hydro.index[0]:BPA_hydro.index[-1]]
#
#Temp_Wind_y=df_weather.resample('D').sum()
#Temp_Wind_y=Temp_Wind_y.drop(index=pd.DatetimeIndex(data=(t for t in dates if isleap(t.year))))
#Temp_Wind_y=Temp_Wind_y.groupby(Calendar_year).max()


############ PLOTS ################################################

#Net revenue Bar plot
plt.rcParams.update({'font.size': 18})

plt.figure()
ax1 = plt.subplot()
ax1 = Net_rev['Net_Rev'].plot(kind="bar",
                      linewidth=0,
                      ax=ax1, color=Net_rev.positive.map({True:'blue', False:'red'}))  # make bar plots
ax1.set_xticklabels(Net_rev.index, rotation = 0)
ax1.set_title('Yearly Net Revenue')
ax1.xaxis.set_ticks(np.arange(1, 1188, 20))
#ax1.set_xticklabels([i for i in range(1,1200,59)])
ax1.set_xticklabels([],[])
ax1.set_yticklabels([i for i in np.arange(-1,2,0.5)])
ax1.set_ylabel('B$')
ax1.grid(linestyle='-', linewidth=0.2)
#axbis = ax1.twinx() 
#axbis.plot(TDA_y, 'steelblue')
#axbis.set_yticks([], [])
#plt.xlim(-1.7 ,20)
plt.tight_layout()
plt.savefig('figures/NetRev1200' + redux)


## Draw the density plot
plt.figure()
ax_pdf=sns.kdeplot(pow(10,-6)*Net_rev['Net_Rev'], shade=True)
# Plot formatting
ax_pdf.legend().set_visible(False)
plt.title('Yearly Net Revenue')
plt.xlabel('$MIllion per year')
ax_pdf.set_ylabel('density')
line = ax_pdf.get_lines()[-1]
x, y = line.get_data()
mask = x < 0
x, y = x[mask], y[mask]
ax_pdf.fill_between(x, y1=y, alpha=0.5, facecolor='red')
ax_pdf.ticklabel_format(style='sci', axis='y', scilimits=(-3,0))
#plt.text(0.5,1.5, 'mean=$M'+str(round(pow(10,-6)*Net_rev['Net_Rev'].mean()))\
#         +'\n'+'std=$M'+str(pow(10,-6)*round(Net_rev['Net_Rev'].std())))
ax_pdf.set_xlim(-850,700)
#ax_pdf.set_ylim(0,3)
plt.show()
plt.savefig('figures/Rev_PDF' + redux, format='eps')


#Calculate VaR
#sort the net revs
Net_rev_sorted=Net_rev['Net_Rev'].sort_values(ascending=True)
Net_rev_sorted.reset_index(drop=True, inplace=True)
VaR_90 = Net_rev_sorted.quantile(0.1)
VaR_95 = Net_rev_sorted.quantile(0.05)
VaR_99 = Net_rev_sorted.quantile(0.01)
from tabulate import tabulate
print (tabulate([['90%', VaR_90],['95%', VaR_95], ['99%', VaR_99]], headers=['Confidence Level', 'Value at Risk']))
plt.axvline(x=VaR_90*pow(10,-6),color= 'yellow')
plt.text(VaR_90*pow(10,-6),1.5*pow(10,-3) , "VaR 90 %d" % VaR_90, rotation=90, verticalalignment='center')
plt.axvline(x=VaR_95*pow(10,-6),color= 'orange')
plt.text(VaR_95*pow(10,-6),1.5*pow(10,-3) , "VaR 95 %d" % VaR_95, rotation=90, verticalalignment='center')
plt.axvline(x=VaR_99*pow(10,-6),color= 'red')
plt.text(VaR_99*pow(10,-6),1.5*pow(10,-3) , "VaR 99 %d" % VaR_99, rotation=90, verticalalignment='center')


idx=np.where(np.diff(np.sign(Net_rev_sorted)))[0]
Negative_percent = 100*((idx+1)/len(Net_rev_sorted))
print ('Percent of negative net revs: %.2f' % Negative_percent )
plt.text(-700,1.5*pow(10,-3) , "perc negatives %f" % Negative_percent, rotation=90, verticalalignment='center')


Net_rev_avg=Net_rev['Net_Rev'].mean()
print('Average Net Revenue: %.2f' % Net_rev_avg)
plt.axvline(x=Net_rev_avg*pow(10,-6))
plt.text(Net_rev_avg*pow(10,-6),1.5*pow(10,-3) , "Average %d" % Net_rev_avg, rotation=90, verticalalignment='center')
plt.savefig('figures/Rev_PDF_lines' + redux + '.eps', format='eps')
plt.savefig('figures/Rev_PDF_lines' + redux, format='png')

#####################################################################
#### ENSEMBLE ANALYSIS ##############

#Create single ensemble horizonatal panels plot 
plt.rcParams.update({'font.size': 12})
for e in range (1,60):
    Net_rev_e=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7])['Net_Rev']
    Positive=Net_rev_e>0
    fig, axes = plt.subplots(nrows=4, ncols=1)
    ax1=axes[0]
    Net_rev_e.plot(kind="bar",
                                          linewidth=0.2,
                                          ax=ax1, 
                                          color=Positive.map({True:'blue', False:'red'}))  # make bar plots
    ax1.set_title('Net Revenue Ensemble '+str(e), pad=0.6)
    ax1.xaxis.set_ticks(range(1, 21, 1))
    ax1.set_xticklabels([],[])
    #ax1.set_xticklabels([i for i in np.arange(1,21,1)])
    ax1.set_ylabel('B$')
    ax1.set_xlim(-0.5,19.5)
    ax1.grid(linestyle='-', linewidth=0.2, axis='x')
    ax1.get_yaxis().set_label_coords(-0.08,0.5)
    
    Reserves_e=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[1])
    Reserves_e=Reserves_e.append(pd.Series(Reserves_e.iloc[19]))
    Reserves_e.reset_index(inplace=True, drop=True)
    Treas_fac1=320*pow(10,-3)   # Treasury facility (1)
    ax2 = axes[1]
    ax2.axhline(0.608691000-Treas_fac1, color='r') 
    ax2.axhline(0, color='r') 
    ax2.plot(Reserves_e ) 
    ax2.set_title('Reserves', pad=0.6)
    ax2.xaxis.set_ticks(range(1, 21, 1))
    ax2.set_xticklabels([],[])
    #ax2.set_xticklabels([i for i in np.arange(1,21,1)])
    ax2.set_ylabel('B$')
    ax2.set_xlim(0.5,20.5)
    ax2.grid(linestyle='-', linewidth=0.2, axis='x')
    ax2.get_yaxis().set_label_coords(-0.08,0.5)

    
    TF1=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[4])
    TF2=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[5])
    TF_e=pd.concat([TF1, TF2], axis=1)
    TF_e.append(TF_e.iloc[19,:])
    TF_e.reset_index(inplace=True, drop=True)
    ax3 = axes[2]
    TF_e.plot(ax=ax3, kind='bar', stacked=True, color=['g','y'],  linewidth=0.2)
    ax3.set_title('Treasury Facility', pad=0.6)
    ax3.set_xticklabels([],[])
    #ax3.set_xticklabels([i for i in np.arange(1,21,1)])
    ax3.set_ylabel('B$')
    ax3.xaxis.set_ticks(range(1, 21, 1))
    ax3.set_ylabel('B$')
    ax3.set_xlim(0.5,20.5)
    ax3.grid(linestyle='-', linewidth=0.2, axis='x')
    ax3.get_yaxis().set_label_coords(-0.08,0.5)
    
    CRAC_e=pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])
    CRAC_e=CRAC_e.append(pd.Series(CRAC_e.iloc[19]))
    CRAC_e.reset_index(inplace=True, drop=True)
    ax4 = axes[3]
    #plot percent increase
    #ax4.plot(CRAC_e*100/PF_rates_avg, 'darkviolet')
    #plot $/MWh increase
    ax4.plot(CRAC_e, 'darkviolet')
    ax4.set_title('Surcharge', pad=0.6)
    ax4.xaxis.set_ticks(range(1, 21, 1))
    ax4.set_xticklabels([i for i in np.arange(1,21,1)])
    #ax4.set_ylabel('%')
    ax4.set_ylabel('$/MWh')
    ax4.set_xlim(0.5,20.5)
    ax4.grid(linestyle='-', linewidth=0.2, axis='x')
    ax4.get_yaxis().set_label_coords(-0.08,0.5)
    
    plt.subplots_adjust(left=0.11, bottom=0.065, right=0.985, top=0.945, wspace=0.2, hspace=0.345)
    plt.savefig('figures/ensembles/Ensembles'+ redux + '/Ensemble'+ str(e))

########### QuantilePlots
# CRAC distribution    
CRAC_e=pd.DataFrame()
for e in range (1,60):
    CRAC_e=pd.concat([CRAC_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])], axis=1)
#Qc=(100/PF_rates_avg)*CRAC_e.T
Qc=CRAC_e.T
Qc.reset_index(inplace=True, drop=True)

#CRAC distribution
count=np.sum(CRAC_e.any())
percent1=100*count/59  #BAU=11.86% 
print ('Percent of CRAC ensembles: %.2f' % percent1 )

#Reserves ensembles
Reserves_e=pd.DataFrame()
for e in range (1,60):
    Reserves_e=(pd.concat([Reserves_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[1])['Reserves'] - 
                           pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[4])['TF1']-
                           pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[5])['TF2'] ], axis=1)) 
Qr=pow(10,-9)*Reserves_e.T
Qr.reset_index(inplace=True, drop=True)

#Revenues ensembles
Revs_e=pd.DataFrame()
for e in range (1,60):
    Revs_e=pd.concat([Revs_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7])['Net_Rev']], axis=1)
Qrev=pow(10,-9)*Revs_e.T
Qrev.reset_index(inplace=True, drop=True)


TTP_e=pd.DataFrame()
for e in range (1,60):
    TTP_e=pd.concat([TTP_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[2])], axis=1)
count=sum(-TTP_e.any()) #0% for both BAU and minus 10% and minus20%

    
## QuantilePlot ensembles function
def quantileplot(Q, ax, color, ci, name, start_day, end_day, realization, tick_interval, log):
    # plot a selected streamflow realization (realization arg) over the
    # quantiles of all streamflow realizations
    if log:
        Q = np.log10(Q)
    ps = np.arange(0,1.01,0.05)*100
    for j in range(1,len(ps)):
        u = np.percentile(Q.iloc[:, start_day:end_day], ps[j], axis=0)
        l = np.percentile(Q.iloc[:, start_day:end_day], ps[j-1], axis=0)
        if ax == ax1:
            ax.fill_between(np.arange(0,len(Q.iloc[0,start_day:end_day])), l, u, \
                            color=cm.twilight_shifted(ps[j-1]/100.0), alpha=0.75, edgecolor='none', label=[str(int(ps[j-1]))+'% to '+ str(int(ps[j])) +'%'])
                            #color=cm.PuOr(ps[j-1]/100.0), alpha=0.75, edgecolor='none')
        else:
            ax.fill_between(np.arange(0,len(Q.iloc[0,start_day:end_day])), l, u, \
                            color=cm.GnBu(ps[j-1]/100.0), alpha=0.75, edgecolor='none',  label=[str(int(ps[j-1]))+'% to '+ str(int(ps[j])) +'%'])
                #color=cm.RdYlBu_r(ps[j-1]/100.0), alpha=0.75, edgecolor='none')
                
    ax.set_xlim([0, end_day-start_day])
    ax.set_xticks(np.arange(0, end_day-start_day+tick_interval, tick_interval))
    ax.set_xticklabels(np.arange(start_day+1, end_day+tick_interval, tick_interval))

    ax.plot(np.arange(0,len(Q.iloc[0,start_day:end_day])), Q.median(), color='k', linewidth=2, label='median')
    #ax.plot(np.arange(0,len(Q.iloc[0,start_day:end_day])), Q.iloc[(realization-1), \
    #    start_day:end_day], color='k', linewidth=2)
    #ax.set_ylim([0, 5])
    #ax.set_yticks(np.arange(6))
    #ax.set_yticklabels([0, '', '', '', '', 5])
    #ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Jan', 'Apr', 'Jul', 'Oct'])

    ax.set_ylabel(name, fontsize=12)
    #ax.set_xlabel('Simulation Day')
    #for xl,yl in zip(ax.get_xgridlines(), ax.get_ygridlines()):
    #   xl.set_linewidth(0.5)
    #    yl.set_linewidth(0.5)
    plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.075, right=0.82, top=0.96, bottom=0.055)
    

fig, ax1 = plt.subplots(1, 1)
quantileplot(Qr, color='k', ax=ax1, ci=90, name='B$', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Net Reserves', size=15)
plt.xlim(0,19)
plt.ylim(-0.67,0.35)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/Reserves' + redux, format='eps')
plt.savefig('figures/Ensembles/Reserves'  + redux  )


fig, ax1 = plt.subplots(1, 1)
quantileplot(Qrev, color='k', ax=ax1, ci=90, name='B$', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Net Revenue', size=15) 
plt.xlim(0,19)
plt.ylim(-0.8, 0.8)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/Net_Rev' + redux, format='eps')


fig, ax1 = plt.subplots(1, 1)
quantileplot(Qc, color='k', ax=ax1, ci=90, name='$/MWh', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Rate increase', size=15) 
plt.xlim(0,19)
plt.ylim(0,6)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/CRAC'  + redux , format='eps')



#####################################################################
#### CORRELATION ANALYSIS ##############

# Load Streamflows
df_streamflow=pd.read_csv('../../CAPOW/CAPOW_SD/Stochastic_engine/Synthetic_streamflows/synthetic_streamflows_FCRPS.csv', header=None)
#cut to fit
df_streamflow = df_streamflow.iloc[365:len(df_streamflow)-(2*365),:]
df_streamflow.reset_index(inplace=True, drop=True)

##Total daily streamflow
FCRPS_d=pd.DataFrame(df_streamflow.sum(axis=1),columns=['FCRPS']).loc[0:365*1200 -1]
#remove missing years 
FCRPS_d=pd.DataFrame(np.reshape(FCRPS_d.values, (365,1200), order='F'))
FCRPS_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
FCRPS_d=pd.DataFrame(np.reshape(FCRPS_d.values, (365*1188), order='F'))
#Cut from October (Water years)
#FCRPS_d=FCRPS_d[273:-92]
#FCRPS_d.reset_index(drop=True,inplace=True)

#same with The Dalles
TDA_d=pd.DataFrame(df_streamflow[47].values,columns=['TDA']).loc[0:365*1200 -1]
TDA_d=pd.DataFrame(np.reshape(TDA_d.values, (365,1200), order='F'))
TDA_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
TDA_d=pd.DataFrame(np.reshape(TDA_d.values, (365*1188), order='F'))
#Cut from October (Water years)
#TDA_d=TDA_d[273:-92]
#TDA_d.reset_index(drop=True,inplace=True)

#total yearly streamflows
FCRPS_y=FCRPS_d.groupby(Calendar_year).sum()
FCRPS_y.reset_index(inplace=True, drop=True)
TDA_y=TDA_d.groupby(Calendar_year).sum()
TDA_y.reset_index(inplace=True, drop=True)

#Other reservoirs:
BON_d=pd.DataFrame(df_streamflow[42].values,columns=['BON']).loc[0:365*1200 -1]
BON_d=pd.DataFrame(np.reshape(BON_d.values, (365,1200), order='F'))
BON_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
BON_d=pd.DataFrame(np.reshape(BON_d.values, (365*1188), order='F'))
BON_y=BON_d.groupby(Calendar_year).sum()
BON_y.reset_index(inplace=True, drop=True)

CHJ_d=pd.DataFrame(df_streamflow[28].values,columns=['CHJ']).loc[0:365*1200 -1]
CHJ_d=pd.DataFrame(np.reshape(CHJ_d.values, (365,1200), order='F'))
CHJ_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
CHJ_d=pd.DataFrame(np.reshape(CHJ_d.values, (365*1188), order='F'))
CHJ_y=CHJ_d.groupby(Calendar_year).sum()
CHJ_y.reset_index(inplace=True, drop=True)

GCL_d=pd.DataFrame(df_streamflow[9].values,columns=['GCL']).loc[0:365*1200 -1]
GCL_d=pd.DataFrame(np.reshape(GCL_d.values, (365,1200), order='F'))
GCL_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
GCL_d=pd.DataFrame(np.reshape(GCL_d.values, (365*1188), order='F'))
GCL_y=GCL_d.groupby(Calendar_year).sum()
GCL_y.reset_index(inplace=True, drop=True)

JDA_d=pd.DataFrame(df_streamflow[40].values,columns=['JDA']).loc[0:365*1200 -1]
JDA_d=pd.DataFrame(np.reshape(JDA_d.values, (365,1200), order='F'))
JDA_d.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
JDA_d=pd.DataFrame(np.reshape(JDA_d.values, (365*1188), order='F'))
JDA_y=JDA_d.groupby(Calendar_year).sum()
JDA_y.reset_index(inplace=True, drop=True)



Net_rev=Net_rev['Net_Rev']
Net_rev.reset_index(inplace=True, drop=True)
TDA_y.reset_index(inplace=True, drop=True)
CHJ_y.reset_index(inplace=True, drop=True)
BON_y.reset_index(inplace=True, drop=True)
PF_load_y=PF_load_y[1:]
PF_load_y.reset_index(inplace=True, drop=True)
IP_load_y=IP_load_y[1:]
IP_load_y.reset_index(inplace=True, drop=True)
MidC_y=MidC_y[1:]
MidC_y.reset_index(inplace=True, drop=True)
CAISO_y=CAISO_y[1:]
CAISO_y.reset_index(inplace=True, drop=True)
FCRPS_y.reset_index(inplace=True, drop=True)


M=pd.concat([Net_rev, TDA_y,CHJ_y,BON_y, GCL_y,JDA_y,PF_load_y,
             IP_load_y, MidC_y, CAISO_y, FCRPS_y],axis=1)
M.columns=['net rev','TDA','CHJ','BON','GCL','JDA','PF_load' ,'IP load','MidC','CAISO','FCRPS']
Corr=M.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(M.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(M.columns)
ax.set_yticklabels(M.columns)
plt.show()


M=pd.concat([Net_rev, TDA_y, GCL_y,MidC_y, CAISO_y, FCRPS_y],axis=1)
M.columns=['net rev','TDA','GCL','MidC','CAISO','FCRPS']
Corr=M.corr().loc[['net rev'],:]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(M.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks([])
ax.set_xticklabels(M.columns)
ax.set_yticklabels('net rev')
for (i, j), z in np.ndenumerate(Corr):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
plt.show()
plt.savefig('figures/correlation/corr' + redux)


data=pd.concat([Net_rev,TDA_y], axis=1)
data.columns=['Net Rev','TDA']
h=sns.jointplot(y='Net Rev',x='TDA', data=data, kind='reg',stat_func=r2)

data=pd.concat([Net_rev,FCRPS_y], axis=1)
data.columns=['Net Rev','Hydrology']
h=sns.jointplot(y='Net Rev',x='Hydrology', data=data, kind='reg',stat_func=r2)

#Negatives=Net_rev.loc[Net_rev['negative'],'Net_Rev']
#data=pd.concat([Negatives, FCRPS_y.loc[Negatives.index]], axis=1)
#data.columns=['Net Rev','hydro']
#h=sns.jointplot(y='Net Rev',x='hydro', data=data, kind='reg',stat_func=r2)
#
#data=pd.concat([Net_rev['Net_Rev'], FCRPS_y, Net_rev['Net_Rev']<Net_rev['Net_Rev'].quantile(0.5)], axis=1)
#data.columns=['Net Rev','hydro', 'low Rev']
#h=sns.lmplot(y='Net Rev',x='hydro', hue='low Rev', data=data)


###MULTIVARIATE OLS
# Prepare price input matrix:
MidC_reshaped=pd.DataFrame(data=np.reshape(Results_d.MidC,(365,1188),order='F'))
CAISO_reshaped=pd.DataFrame(data=np.reshape(Results_d.CAISO,(365,1188),order='F'))
Load_reshaped=pd.DataFrame(data=np.reshape(Results_d.PF_load,(365,1188),order='F'))
FCRPS_d=pd.DataFrame(np.reshape(FCRPS_d.values, (365,1188), order='F'))
TDA_d=pd.DataFrame(np.reshape(TDA_d.values, (365,1188), order='F'))
Rev_d=pd.DataFrame(np.reshape(Results_d.Rev_gross, (365,1188), order='F'))

##deviations
#FCRPS_devs1=(FCRPS_d.T-FCRPS_d.mean(axis=1)).max(axis=1)
#FCRPS_devs2=(FCRPS_d.T-FCRPS_d.mean(axis=1)).mean(axis=1)
#FCRPS_devs3=(FCRPS_d.T-FCRPS_d.mean(axis=1)).min(axis=1)
#
#Load_devs1=(Load_reshaped.T-Load_reshaped.mean(axis=1)).max(axis=1)
#Load_devs2=(Load_reshaped.T-Load_reshaped.mean(axis=1)).mean(axis=1)
#Load_devs3=(Load_reshaped.T-Load_reshaped.mean(axis=1)).min(axis=1)
#
#MidC_devs1=(MidC_reshaped.T-MidC_reshaped.mean(axis=1)).max(axis=1)
#MidC_devs2=(MidC_reshaped.T-MidC_reshaped.mean(axis=1)).mean(axis=1)
#MidC_devs3=(MidC_reshaped.T-MidC_reshaped.mean(axis=1)).min(axis=1)
#
#CAISO_devs1=(CAISO_reshaped.T-CAISO_reshaped.mean(axis=1)).max(axis=1)
#CAISO_devs2=(CAISO_reshaped.T-CAISO_reshaped.mean(axis=1)).mean(axis=1)
#CAISO_devs3=(CAISO_reshaped.T-CAISO_reshaped.mean(axis=1)).min(axis=1)


# Early snowmelt season MAR-APR (APR-JUN)
FCRPS_spring=FCRPS_d.iloc[60:120,:].mean(axis=0)
TDA_spring=TDA_d.iloc[60:120,:].mean(axis=0)
Load_spring=Load_reshaped.iloc[60:120,:].mean(axis=0)
MidC_spring=MidC_reshaped.iloc[60:120,:].mean(axis=0)
CAISO_spring=CAISO_reshaped.iloc[60:120,:].mean(axis=0)
Rev_spring=Rev_d.iloc[60:120,:].mean(axis=0)

M=pd.concat([Rev_spring,FCRPS_spring, TDA_spring, Load_spring, MidC_spring, CAISO_spring],axis=1)
M.columns=['REV','hydrology', 'TDA', 'Load','MidC','CAISO']
Corr=M.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(M.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(M.columns)
ax.set_yticklabels(M.columns)
plt.title('Spring MAR-APR')
plt.show()


# Late Summer (AUG-SEPT)
FCRPS_summer=FCRPS_d.iloc[213:273,:].mean(axis=0)
TDA_summer=TDA_d.iloc[213:273,:].mean(axis=0)
Load_summer=Load_reshaped.iloc[213:273,:].mean(axis=0)
MidC_summer=MidC_reshaped.iloc[213:273,:].mean(axis=0)
CAISO_summer=CAISO_reshaped.iloc[213:273,:].mean(axis=0)
Rev_summer=Rev_d.iloc[213:273,:].mean(axis=0)


M=pd.concat([Rev_summer,FCRPS_spring, TDA_spring, Load_spring, MidC_spring, CAISO_spring],axis=1)
M.columns=['REV','hydrology', 'TDA', 'Load','MidC','CAISO']
Corr=M.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(M.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(M.columns)
ax.set_yticklabels(M.columns)
plt.title('Summer AUG-SEP')
plt.show()

# Hydrology and prices during the winter DEC and JAN
FCRPS_winter=pd.concat([FCRPS_d.iloc[0:30,:],FCRPS_d.iloc[334:364,:]],axis=0).mean(axis=0)
TDA_winter=pd.concat([TDA_d.iloc[0:30,:],TDA_d.iloc[334:364,:]],axis=0).mean(axis=0)
Load_winter=pd.concat([Load_reshaped.iloc[0:30,:],Load_reshaped.iloc[334:364,:]],axis=0).mean(axis=0)
MidC_winter=pd.concat([MidC_reshaped.iloc[0:30,:],MidC_reshaped.iloc[334:364,:]],axis=0).mean(axis=0)
CAISO_winter=pd.concat([CAISO_reshaped.iloc[0:30,:],CAISO_reshaped.iloc[334:364,:]],axis=0).mean(axis=0)
Rev_winter=pd.concat([Rev_d.iloc[0:30,:],Rev_d.iloc[334:364,:]],axis=0).mean(axis=0)

M=pd.concat([Rev_winter,FCRPS_spring, TDA_spring, Load_spring, MidC_spring, CAISO_spring],axis=1)
M.columns=['REV','hydrology', 'TDA', 'Load','MidC','CAISO']
Corr=M.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(Corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(M.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(M.columns)
ax.set_yticklabels(M.columns)
plt.show()


## REGRESSION ANALYSIS
X = pd.concat([FCRPS_y, TDA_y,  MidC_spring], axis=1).loc[0:1179,:]
X.columns=['hydrology','TDA','MidC']
#X = pd.concat([FCRPS_y, TDA_y,GCL_y, CAISO_summer, MidC_spring], axis=1)
#X.columns=['hydrology','TDA','GCL','CAISO','MidC']
y=Net_rev
df=pd.concat([X,y], axis=1)

for i in df.columns:
    df.plot.scatter(i,'Net_Rev', edgecolors=(0,0,0),s=50,c='g',grid=True)
    #plt.savefig('figures/correlation/scatter_'+ i)

#LINEAR    
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize=True)
linear_model.fit(X,y)
y_pred = linear_model.predict(X)
RMSE = np.sqrt(np.sum(np.square(y_pred-y)))
print("Root-mean-square error of linear model:",RMSE)
coeff_linear = pd.DataFrame(linear_model.coef_,index=X.columns, columns=['Linear model coefficients'])
coeff_linear
print ("R2 value of linear model:",linear_model.score(X,y))

plt.figure()
plt.xlabel("Predicted Net Revenues",fontsize=20)
plt.ylabel("Actual Net Revenues",fontsize=20)
plt.grid(1)
plt.scatter(y_pred,y,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(y_pred,y_pred, 'k--', lw=2)
plt.text(-2*pow(10,8),0.6*pow(10,9),'R2='+str('{:0.2f}'.format(linear_model.score(X,y))))
plt.title('OLS model', size=18)
plt.savefig('figures/OLS_model' + redux)

#POLY
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2,include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_feature_name = poly.get_feature_names(X.columns)
print(X_poly_feature_name)
print(len(X_poly_feature_name))

df_poly = pd.DataFrame(X_poly, columns=X_poly_feature_name)
df_poly.head()


df_poly['y']=df['Net_Rev']
df_poly.head()


X_train=df_poly.drop('y',axis=1)
y_train=df_poly['y']

poly = LinearRegression(normalize=True)
model_poly=poly.fit(X_train,y_train)
y_poly = poly.predict(X_train)
RMSE_poly=np.sqrt(np.sum(np.square(y_poly-y_train)))
print("Root-mean-square error of simple polynomial model:",RMSE_poly)

coeff_poly = pd.DataFrame(model_poly.coef_,index=df_poly.drop('y',axis=1).columns, 
                          columns=['Coefficients polynomial model'])
coeff_poly

print ("R2 value of simple polynomial model:",model_poly.score(X_train,y_train))

#Metamodel: polynomial model with cross-validation and LASSO regularization
#Lasso
from sklearn.linear_model import LassoCV
model1 = LassoCV(cv=10,verbose=0,normalize=True,eps=0.001,n_alphas=100, tol=0.0001,max_iter=5000)
model1.fit(X_train,y_train)
y_pred1 = np.array(model1.predict(X_train))
RMSE_1=np.sqrt(np.sum(np.square(y_pred1-y_train)))
print("Root-mean-square error of Metamodel:",RMSE_1)
model1.score(X_train,y_train)

plt.figure()
plt.xlabel("Predicted Net Revenues",fontsize=20)
plt.ylabel("Actual Net Revenues",fontsize=20)
plt.grid(1)
plt.scatter(y_pred1,y_train,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(y_pred1,y_pred1, 'k--', lw=2)
plt.text(-0.6*pow(10,9),0.6*pow(10,9),'R2='+str('{:0.2f}'.format(model1.score(X_train,y_train))))
plt.title('2nd degree polynomial model with cross-validation and LASSO regularization', size=18)
plt.savefig('figures/Poly_model' + redux)


#### VALIDATION PLOT
df_valid=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=6, usecols=np.arange(12,15))


g = sns.catplot(x='year', y='Values', hue="type", data=df_valid,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("$M")
plt.savefig('figures/Validation', format='eps')

df_valid2=pd.read_excel('../DATA/net_rev_data.xlsx',sheet_name=7)

plt.figure()
p = sns.regplot(x='historical', y='simulated', data=df_valid2)
