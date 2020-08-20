#!/usr/bin/env python
# coding: utf-8

# # Data Exploration of Scene Data and Preliminary Data Analysis

# In[1]:


import pandas as pd

import os

import pandas_profiling

import matplotlib.pyplot as plt

import numpy as np

import featuretools as ft

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## Read SP_Points

# In[2]:


SP_Points = pd.read_csv("SceneAnalytics.dbo.SP_Points_cleaned.csv", encoding='latin-1', parse_dates=['pointdt'])

SP_Points['isBlackCard'] = SP_Points['pointtypeid'].apply(lambda x: True if x in [11,12,1252,1253,1254,1282,1283,1290,1322,1323] else False)

SP_PointsType =pd.read_csv("SceneAnalytics.dbo.SP_PointsType.csv", encoding='latin-1')
SP_LocationCARA =pd.read_csv("SceneAnalytics.dbo.SP_LocationCARA.csv", encoding='latin-1')
SP_Location =pd.read_csv("SceneAnalytics.dbo.SP_Location.csv", encoding='latin-1')
SP_Partner_E =pd.read_csv("SceneAnalytics.dbo.SP_Partner_E.csv", encoding='latin-1')

SP_Points = SP_Points.merge(SP_PointsType, left_on="ex_sourceid", right_on="pointtypeid", how="left", suffixes=('', 'SP_PointsType'))
SP_Points = SP_Points.merge(SP_LocationCARA, left_on="ex_sourceid", right_on="Locationcode", how="left", suffixes=('', 'SP_LocationCARA'))
SP_Points = SP_Points.merge(SP_Location, left_on="ex_sourceid", right_on="LocationCode", how="left", suffixes=('', 'SP_Location'))
SP_Points = SP_Points.merge(SP_Partner_E, left_on="ex_sourceid", right_on="PartnerID", how="left", suffixes=('', 'SP_Partner_E'))


#profile = pandas_profiling.ProfileReport(SP_Points, html={'style':{'full_width':True}}, minimal=True)
#profile.to_notebook_iframe()


# In[3]:


SP_Points['Unique_member_identifier'].value_counts().head()


# ## Read SP_CustomerDetail

# In[4]:


SP_CustomerDetail = pd.read_csv("SceneAnalytics.dbo.SP_CustomerDetail_cleaned.csv", encoding='latin-1')
SP_PointTypeStatistics = pd.read_csv("SceneAnalytics.dbo.SP_PointTypeStatistics.csv", encoding='latin-1', 
                                     parse_dates=['BlackEarnLastDt', 'BlackBurnLastDt', 'LoadTime', 'OrderLastDt', 'LastDt', 'ChildTicketLastDt',
                                                  'ConcessionLastDt', 'CnplxEarnTuesdayLastDt', 'MusicStoreLastDt', 'CnplxOnlineBonusLastDt'])
SP_FactEnrollment = pd.read_csv("SceneAnalytics.dbo.SP_FactEnrollment.csv", encoding='latin-1', parse_dates=['Birthdate', 'loadtime'])
SP_FactAttribute = pd.read_csv("SceneAnalytics.dbo.SP_FactAttribute.csv", encoding='latin-1')
SP_AccountBalance = pd.read_csv("SceneAnalytics.dbo.SP_AccountBalance.csv", encoding='latin-1')
SP_CustomerExtension = pd.read_csv("SceneAnalytics.dbo.SP_CustomerExtension.csv", encoding='latin-1', parse_dates=['CreateDt'])


# In[5]:


SP_Customer = SP_CustomerDetail

SP_Customer = SP_Customer.merge(SP_PointTypeStatistics, how="left", suffixes=('', 'SP_PointTypeStatistics'))
SP_Customer = SP_Customer.merge(SP_FactEnrollment, how="left", suffixes=('', 'SP_FactEnrollment'))
SP_Customer = SP_Customer.merge(SP_FactAttribute, how="left", suffixes=('', 'SP_FactAttribute'))
SP_Customer = SP_Customer.merge(SP_AccountBalance, on="Unique_member_identifier", how="left", suffixes=('', 'SP_AccountBalance'))
SP_Customer = SP_Customer.merge(SP_CustomerExtension, how="left", suffixes=('', 'SP_CustomerExtension'))


# In[6]:


#profile = pandas_profiling.ProfileReport(SP_Customer, html={'style':{'full_width':True}}, minimal=True)
#profile.to_notebook_iframe()


# In[7]:


SP_CustomerDetail['Unique_member_identifier'].value_counts().head()


# In[8]:


# https://stackoverflow.com/questions/27637281/what-are-python-pandas-equivalents-for-r-functions-like-str-summary-and-he
#https://gist.github.com/minhchan11/4e2f80383f4e93e764308776d116580a
def rstr(df): 
    structural_info = pd.DataFrame(index=df.columns)
    structural_info['unique_len'] = df.apply(lambda x: len(x.unique())).values
    structural_info['unique_val'] = df.apply(lambda x: [x.unique()]).values
    print(df.shape)
    return structural_info 


# ## Head & Shape
# How do we filter the Points table for the transactions specifically related to their spend within theatres?

# In[9]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[10]:


SP_Customer.head()


# In[11]:


SP_Customer.shape


# In[12]:


SP_Points.head()


# In[13]:


SP_Points.shape


# Purchases made on black card?

# In[14]:


SP_Points[(SP_Points['isBlackCard']==True)].shape


# ## isBlackCard

# In[15]:


SP_Points[(SP_Points['isBlackCard']==True)].head(15)


# In[16]:


SP_Points[(SP_Points['isBlackCard']==False)].head(15)


# How do we filter for in theatre purchases? From below we can see that this includes step_goals, cara points, etc. How do we limit this to spending? The hypothesis would be that we can use LocationNameSP_Location not null to do so.

# ### Transaction Descriptions for Black Card = True

# In[17]:


transaction_counts_true = SP_Points[SP_Points['isBlackCard']==True].groupby(['ex_transactiondescription', 'isBlackCard'])[['Unique_member_identifier']].count().reset_index()
transaction_counts_true.sort_values(by=['Unique_member_identifier'], ascending=False).head(100)


# ### Transaction Descriptions for Black Card = False

# In[18]:


transaction_counts_false = SP_Points[SP_Points['isBlackCard']==False].groupby(['ex_transactiondescription', 'isBlackCard'])[['Unique_member_identifier']].count().reset_index()
transaction_counts_false.sort_values(by=['Unique_member_identifier'], ascending=False).head(10)


# Looks like the location hypothesis is true. Some of these have no transaction amounts.

# ### Transaction Descriptions when Location Name is Not Null
# Location Name narrows this down to Cineplex locations

# In[19]:


tcounts_loc = SP_Points[(SP_Points['isBlackCard']==True) & SP_Points['LocationNameSP_Location'].notnull()].groupby(['ex_transactiondescription', 'isBlackCard'])[['Unique_member_identifier']].count().reset_index()


# In[20]:


tcounts_loc.sort_values(by=['Unique_member_identifier'], ascending=False).head(100)


# ### Transaction Descriptions when Location Name is Null
# Location Name blank are Black Card transactions associated with Carrot and Sport Chek partnerships at the time

# In[21]:


tcounts_loc_null = SP_Points[(SP_Points['isBlackCard']==True) & SP_Points['LocationNameSP_Location'].isnull()].groupby(['ex_transactiondescription', 'isBlackCard', 'PartnerName'])[['Unique_member_identifier']].count().reset_index()


# In[22]:


tcounts_loc_null.sort_values(by=['Unique_member_identifier'], ascending=False).head(100)


# ### BlackCard = False, Cineplex Visa/Debit Transactions

# In[23]:


SP_Points[(SP_Points['ex_transactiondescription']=='cineplex')].head(100)


# ### Timeplay Transactions

# In[24]:


SP_Points[(SP_Points['ex_transactiondescription']=='timeplay')].head()


# ### Misc
# #### Cara Points Earned

# In[25]:


SP_Points[(SP_Points['ex_transactiondescription']=='cara points earned')].head(100)


# #### Unique Values Black Card False, transactions
# How would we measure education and entertainment?
# They measure - what the cluster enjoys doing, and where they hang out.
# Can some of below be encoded as entertainment vs educational? i.e. LCBO
# Can a unique customer be found in both non black card and blackcard? YES
# Where do Scene points come from if not black card? Credit card
# Restaurants - cheap, 
# Add back cineplex
# monthly spend?

# In[26]:


SP_Points[(SP_Points['isBlackCard']==False)]['ex_transactiondescription'].value_counts().head(100)


# #### Unique Values Partner Name

# In[27]:


SP_Points['PartnerName'].value_counts()


# In[28]:


SP_Points[(SP_Points['PartnerName']=='Carrot Rewards')].head(100)


# In[29]:


SP_Points[(SP_Points['PartnerName']=='Sport Chek')].head()


# Unique member is tied to both black card purchases and not

# ### Do members have both VISA/Debit and Scene cards? Yes

# In[30]:


SP_Points.pivot_table(index='Unique_member_identifier', 
               columns='isBlackCard',
               aggfunc='size',
               fill_value=0).head(100)


# In[31]:


SP_Points[(SP_Points['isBlackCard']==False) & SP_Points['LocationNameSP_Location'].notnull()].tail(50)


# ### Does BlackCard have transaction values? Pretty much no
# Will the TransAmounts have actual values?
# https://stackoverflow.com/questions/47170311/pandas-groupby-count-non-null-values-as-percentage

# In[32]:


BlackCard_Cineplex = SP_Points[(SP_Points['isBlackCard']==True) & SP_Points['LocationNameSP_Location'].notnull()]


# In[33]:


BlackCard_Cineplex.set_index("ex_transactiondescription").notnull().groupby(level=0).mean()


# In[34]:


BlackCard_Cineplex[BlackCard_Cineplex['ex_transactiondescription']=='cara points earned']


# # Data Preparation

# ## Creating a new dataframe for movie transactions only
# 
# Need a dataframe that has Customer, Date (no time), Net Points, Actual Spend, Converted Spend, Category [Black Card Only, Visa/Debit Only, Black Card + Visa/Debit)
# 
# First, start by creating 2 dataframes (1 for BC, 1 for non-BC)

# In[35]:


from datetime import datetime


# In[36]:


#https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
SP_Points['pointdt_new'] = SP_Points['pointdt'].apply(lambda x:x.date().strftime('%Y-%m-%d'))


# ### Split SP_Points into Black Card Movie / Non Black Card Movie Transactions

# In[37]:


bc_movie = SP_Points[(SP_Points['isBlackCard']==True) & (SP_Points['LocationNameSP_Location'].notnull())]
non_bc_movie = SP_Points[(SP_Points['isBlackCard']==False) & (SP_Points['ex_transactiondescription']=='cineplex')]


# #### Aggregate the Transactions into Daily Totals

# In[38]:


bc_movie_day = bc_movie[['Unique_member_identifier', 'pointdt_new', 'points', 'TransAmount', 'isBlackCard']]


# In[39]:


bc_movie_day = bc_movie_day.groupby(['Unique_member_identifier', 'pointdt_new', 'isBlackCard']).sum().reset_index()


# In[40]:


non_bc_movie_day = non_bc_movie[['Unique_member_identifier', 'pointdt_new', 'points', 'TransAmount','isBlackCard']]


# In[41]:


non_bc_movie_day = non_bc_movie_day.groupby(['Unique_member_identifier', 'pointdt_new', 'isBlackCard']).sum().reset_index()


# In[42]:


print(bc_movie_day.shape)
print(non_bc_movie_day.shape)


# In[43]:


bc_movie_day.head()


# What are the 100/50 point earn days? These are mostly bonuses for buying tickets.

# ### Pulling up the transactions for 1 member

# In[44]:


SP_Points[(SP_Points['Unique_member_identifier']=='001D64FC-065E-432B-9F8A-4C0075554525') & (SP_Points['isBlackCard']==True)]


# ### Do members ever use both black scene card and visa/debit on the same day? Rarely

# In[45]:


merged_df = bc_movie_day.merge(non_bc_movie_day, how='left', on=['Unique_member_identifier','pointdt_new'], suffixes=['BC','NonBC'])


# In[46]:


merged_df.shape


# In[47]:


merged_df[merged_df.pointsNonBC.notnull()].shape


# In[48]:


(66*100)/42632


# No real need to merge it. Only 0.15% of instances they used both on the same day.
# 

# ### Create a list of transactions, so that I can assign dollar values to these 

# In[49]:


transaction_pt_avgs = SP_Points[(SP_Points['isBlackCard']==True) & SP_Points['LocationNameSP_Location'].notnull()].groupby(['ex_transactiondescription', 'isBlackCard'])[['points']].agg(['mean', 'count']).reset_index()


# In[50]:


transaction_pt_avgs.head()


# In[51]:


#https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
##https://stackoverflow.com/questions/38727863/how-to-access-multiindex-column-after-groupby-in-pandas
transaction_pt_avgs.info()


# In[52]:


transaction_pt_avgs.sort_values(by=[('points','count')], ascending=False).head(100)


# In[53]:


# output to excel for further spreadsheet analysis
#transaction_pt_avgs.sort_values(by=[('points','count')], ascending=False).to_excel("point_earnings.xlsx")


# ## Points To Dollars Conversion

# In[54]:


bc_movie[['TransAmount']] = bc_movie[['TransAmount']].fillna(value=0)


# In[55]:


bc_movie['PointsConverted'] = bc_movie['points']/5


# In[56]:


bc_movie['pointdt_new'] = pd.to_datetime(bc_movie['pointdt_new'])


# In[57]:


non_bc_movie_day['pointdt_new'] = pd.to_datetime(non_bc_movie_day['pointdt_new'])


# In[58]:


def convert_points(row):
    if row['points']<0:
        amount =  0
    elif row['points']>=100:
        if datetime.weekday(row['pointdt_new'])==1: amount = 7.99
        else: amount = 12.99
    elif row['points']>=200:
        if datetime.weekday(row['pointdt_new'])==1: amount = 15.99
        else: amount = 24.50      
    elif row['ex_transactiondescription']=="adult tickets": 
        if datetime.weekday(row['pointdt_new'])==1: amount = 12.99     
        else: amount = 7.99
    elif row['ex_transactiondescription']=="adult tickets premium": 
        if datetime.weekday(row['pointdt_new'])==1: amount = 24.50   
        else: amount = 15.99
    elif row['ex_transactiondescription']=="child tickets":
        if datetime.weekday(row['pointdt_new'])==1: amount = 8.99
        else: amount = 7.99 
    elif row['ex_transactiondescription']=="adult tickets vip": 
        if datetime.weekday(row['pointdt_new'])==1: amount = 19.99
        else: amount = 14.99   
    elif row['ex_transactiondescription']=="timeplay": amount = 0
    elif row['ex_transactiondescription']=="online bonus en ligne": amount = 0
    elif row['ex_transactiondescription']=="xscape items": amount = row['points']/10
    elif row['ex_transactiondescription']=="rec room": amount = row['points']/10
    else:
        amount = row['PointsConverted']
    return amount


# In[59]:


bc_movie['PointsConverted'] = bc_movie.apply(convert_points, axis=1)


# In[60]:


bc_movie.head(20)


# In[61]:


bc_movie['TransAmount'] = bc_movie['TransAmount']+bc_movie['PointsConverted']


# In[62]:


bc_movie_day = bc_movie[['Unique_member_identifier', 'pointdt_new', 'points', 'TransAmount', 'isBlackCard']]


# In[63]:


bc_movie_day = bc_movie_day.groupby(['Unique_member_identifier', 'pointdt_new', 'isBlackCard']).sum().reset_index()


# In[64]:


bc_movie_day.head(20)


# In[65]:


combined_movies = pd.concat([bc_movie_day,non_bc_movie_day])


# In[66]:


combined_movies.shape


# In[67]:


ax2 = combined_movies[['TransAmount']].plot.hist(alpha=0.5, bins=50)
ax2.set_xlim(0,100)


# In[68]:


combined_movies['SpendOver30'] = combined_movies['TransAmount'].apply(lambda x: 1 if x > 30 else 0)


# Reviewing Average Spend on Concessions

# In[69]:


bc_movie[bc_movie['ex_transactiondescription']=="concessions"]['TransAmount'].mean()


# ### Calculating the Number of Months Between Users First and Last Transaction (No Longer Used)
# https://stackoverflow.com/questions/25024797/max-and-min-date-in-pandas-groupby

# In[70]:


minmaxdates = combined_movies.groupby(['Unique_member_identifier']).agg({'pointdt_new': [np.min,np.max]})


# In[71]:


minmaxdates.columns


# In[72]:


minmaxdates.info()


# Rename columns to be a bit more clean
# https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas

# In[73]:


minmaxdates.columns.set_levels(['earliestdate','latestdate'],level=1,inplace=True)


# In[74]:


minmaxdates.head()


# In[75]:


minmaxdates[('pointdt_new','latestdate')] = pd.to_datetime(minmaxdates[('pointdt_new','latestdate')])


# In[76]:


minmaxdates[('pointdt_new','earliestdate')] = pd.to_datetime(minmaxdates[('pointdt_new','earliestdate')])


# In[77]:


# https://stackoverflow.com/questions/42822768/pandas-number-of-months-between-two-dates
minmaxdates[('','user_months')] = ((minmaxdates[('pointdt_new','latestdate')] - minmaxdates[('pointdt_new','earliestdate')])/np.timedelta64(1, 'M'))
#minmaxdates[('','user_months')]  = minmaxdates[('','user_months')] .astype(int)


# In[78]:


import math
minmaxdates[('','user_months')] = minmaxdates[('','user_months')].apply(math.ceil)
# For people who only had 1 transaction
minmaxdates[('','user_months')] = minmaxdates[('','user_months')].apply(lambda x: 1 if x == 0 else x)


# In[79]:


minmaxdates.columns = minmaxdates.columns.droplevel(0)


# In[80]:


minmaxdates = minmaxdates.reset_index()


# In[81]:


# Get User Total Spend and average spend using feature tools
combined_movies.head()


# https://www.youtube.com/watch?v=Q5U9rEKHIsk https://docs.featuretools.com/en/stable/loading_data/using_entitysets.html https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html#featuretools.dfs import featuretools as ft

# ### Revised # of Months

# In the review of the clusters, we determined that we actually need the calculate the total number of months since they got the card, not just the number of months between the first and last movie.

# In[82]:


SP_Customer_Dates = SP_Customer[['Unique_member_identifier','AccountOpenKey','AccountCloseKey']]
SP_Customer_Dates['AccountOpenKey'] = pd.to_datetime(SP_Customer_Dates['AccountOpenKey'], format='%Y%m%d')
SP_Customer_Dates['AccountCloseKey'] = pd.to_datetime(SP_Customer_Dates['AccountCloseKey'], format='%Y%m%d')
SP_Customer_Dates['Last_Transaction_Date'] = pd.to_datetime("20171213", format='%Y%m%d')
SP_Customer_Dates['EitherCloseOrLastDate'] = SP_Customer_Dates.apply(lambda x: x.Last_Transaction_Date if x.AccountCloseKey == pd.Timestamp(2100,1,1) else x.AccountCloseKey, axis=1)
SP_Customer_Dates['user_months'] = ((SP_Customer_Dates['EitherCloseOrLastDate'] - SP_Customer_Dates['AccountOpenKey'])/np.timedelta64(1, 'M'))
SP_Customer_Dates['user_months'] = SP_Customer_Dates['user_months'].apply(math.ceil)
# For people who only had 1 transaction
SP_Customer_Dates['user_months'] = SP_Customer_Dates['user_months'].apply(lambda x: 1 if x == 0 else x)


# In[83]:


SP_Customer_Dates.info()


# In[84]:


MovieCusts = minmaxdates[['Unique_member_identifier']]
SP_Customer_Dates = SP_Customer_Dates.merge(MovieCusts, how='inner')


# In[85]:


SP_Customer_Dates = SP_Customer_Dates[['Unique_member_identifier','user_months']]


# Quick spot check of teh math

# In[86]:


SP_Customer_Dates[SP_Customer_Dates['Unique_member_identifier']=='DAFCC087-5094-462A-9CEA-E3DC6A785835']


# In[87]:


combined_movies[combined_movies['Unique_member_identifier']=='DAFCC087-5094-462A-9CEA-E3DC6A785835']


# ### Aggregating Points/TransAmount from SP_Points

# In[88]:


#min max dates is the customer table. combined_movies is the transactional table.
es = ft.EntitySet(id="customer_spend")
es = es.entity_from_dataframe(entity_id="transactions_new",
                              dataframe=combined_movies,
                              index="trans_id"
                              #,make_index=True
                             )
es = es.entity_from_dataframe(entity_id="customers", dataframe=SP_Customer_Dates, index="Unique_member_identifier")
new_relationship = ft.Relationship(es["customers"]["Unique_member_identifier"], es["transactions_new"]["Unique_member_identifier"])
es = es.add_relationship(new_relationship)
cust_spend, spend_defs = ft.dfs(entityset=es, entities=es, relationships=new_relationship,
                                           target_entity="customers", agg_primitives=['mean', 'sum', 'count'], ignore_variables={"transactions_new": ["index"]})


# In[89]:


spend_defs


# The mean is the average spent per movie. The sum is the total amount spent. The user months is from the customer table and reflects the total number of months between the first and last transaction.

# In[90]:


cust_spend['AvgSpendPerMonth']=cust_spend['SUM(transactions_new.TransAmount)']/cust_spend['user_months']
cust_spend['AvgMoviesPerMonth']=cust_spend['COUNT(transactions_new)']/cust_spend['user_months']


# In[91]:


cust_spend.rename(columns={"MEAN(transactions_new.points)": "AvgNetPointsPerMovie", "MEAN(transactions_new.TransAmount)": "AvgSpendPerMovie",
                                   "SUM(transactions_new.points)": "TtlNetPoints", "SUM(transactions_new.TransAmount)": "TtlMovieSpend",
                          "MEAN(transactions_new.SpendOver30)": "PercentSpendOver30"}, inplace=True)
cust_spend.drop(['user_months', 'COUNT(transactions_new)', 'SUM(transactions_new.SpendOver30)'], axis=1, inplace=True)


# In[92]:


cust_spend.head()


# ## Average Time Between Seeing a Movie
# Code created by Hamza Munir

# In[93]:


avg_time = combined_movies.copy()


# In[94]:


#year of transaction
avg_time['Year_pointdt'] = avg_time['pointdt_new'].astype(str).str[:4]

#format date
avg_time['pointdt_new']=pd.to_datetime(avg_time['pointdt_new'], format='%Y-%m-%d')

#calculate visit number per year
avg_time['visit_nr_yr'] = avg_time.groupby(['Unique_member_identifier', 'Year_pointdt']).cumcount()+1

avg_time = avg_time.sort_values(['Unique_member_identifier', 'pointdt_new'])

avg_time['previous_visit'] = avg_time.groupby(['Unique_member_identifier', 'Year_pointdt'])['pointdt_new'].shift()
avg_time['AvgDaysBetweenMovies'] = avg_time['pointdt_new'] - avg_time['previous_visit']
avg_time['AvgDaysBetweenMovies'] = avg_time['AvgDaysBetweenMovies'].apply(lambda x: x.days)

avg_time = avg_time.groupby('Unique_member_identifier')['AvgDaysBetweenMovies'].agg('mean')


# Some users have only seen 1 movie. Need to calculate the time since last movie

# In[95]:


last_trans_vs_earliest_date = minmaxdates.copy()
last_trans_date = (combined_movies['pointdt_new'].max())


# In[96]:


last_trans_vs_earliest_date.head()


# In[97]:


last_trans_vs_earliest_date['longesttime'] = ((last_trans_date-last_trans_vs_earliest_date['latestdate'])/np.timedelta64(1, 'D')).astype(int)


# In[98]:


last_trans_vs_earliest_date.drop(columns=['earliestdate','latestdate','user_months'], axis=1, inplace=True)


# ## Time Of Day Features
# Files Created by Jaspal Panesar
# 
#     "import datetime
# 
#     combined_movies['hour'] = combined_movies['pointdt'].hour
# 
#     AvgTimeOfDay = combined_movies.groupby(['Unique_member_identifier])['hour'].agg('avg')"

# In[99]:


timeofday = pd.read_csv("TimeofDay.csv", usecols = ['Row Labels','Day','Evening', 'Late Night'])


# In[100]:


timeofday['ModeTime']=timeofday[['Day','Evening', 'Late Night']].idxmax(axis=1)
timeofday.rename(columns={'Row Labels': 'Unique_member_identifier', 'Day': 'NumDayTimeMovies', 'Evening': 'NumEveningMovies',
                         'Late Night': 'NumLateNightMovies'}, inplace=True)
timeofday.head()


# In[101]:


avghour = pd.read_csv("AvgHour.csv")
avghour.rename(columns={'pointdt_hour': 'AvgTimeOfDay'}, inplace=True)


# In[102]:


avghour['AvgTimeOfDay'].plot.hist(bins=24)


# In[103]:


avghour.info()


# In[104]:


avghour.loc[(avghour['AvgTimeOfDay'] <4) & (avghour['AvgTimeOfDay'] >=0),'AvgTimeOfDay'] = 24
avghour.loc[(avghour['AvgTimeOfDay'] >=4) & (avghour['AvgTimeOfDay'] <=10),'AvgTimeOfDay'] = 10


# Dealing with outliers for AvgTimeOfDay

# In[105]:


avghour['AvgTimeOfDay'].plot.hist(bins=24)


# In[ ]:





# ## User Reachable Via SMS or Email
# Files Created by Diana Nwokedi

# In[106]:


emailorsms = pd.read_csv("IsQuality.csv")
emailorsms.rename(columns={'> 85% Avg': 'UserReachableViaSMSEmail'}, inplace=True)


# ## Calculating Birthdate and Urban/Suburban

# We assume the date is the last date

# In[107]:


last_trans_date


# In[108]:


SP_Customer['Age'] = ((last_trans_date-SP_Customer['Birthdate'])/np.timedelta64(1, 'Y')).astype(int)


# We use the FSN to determine if it's Urban or Rural
# https://www.ic.gc.ca/eic/site/bsf-osb.nsf/eng/br03396.html

# In[109]:


SP_Customer['Urban'] = SP_Customer['FSA'].apply(lambda x: 0 if x[1]=="0" else 1)


# In[110]:


SP_Customer['Urban'].value_counts()


# # Carrot and Timeplay Features

# In[111]:


SP_Points[SP_Points['PartnerName']== 'Carrot Rewards']['ex_transactiondescription'].value_counts()


# In[112]:


carrot_timeplay = SP_Points[SP_Points['ex_transactiondescription'].isin(['timeplay','step_goal_complete','intervention'])]


# In[113]:


carrot_timeplay = pd.pivot_table(carrot_timeplay, values='points', index=['Unique_member_identifier'],
                    columns=['ex_transactiondescription'], aggfunc=np.sum, fill_value=0).reset_index()


# In[114]:


carrot_timeplay.rename(columns={'intervention': 'TtlCarrotInterventionPoints', 'step_goal_complete': 'TtlStepPointsCarrot', 'timeplay': 'TtlTimeplay'}, inplace=True)


# In[115]:


carrot_total = SP_Points[SP_Points['PartnerName']== 'Carrot Rewards'][['Unique_member_identifier','points']].groupby('Unique_member_identifier').sum().reset_index()


# In[116]:


emailorsms.rename(columns={'> 85% Avg': 'UserReachableViaSMSEmail'}, inplace=True)


# # Building Customer Base Table

# In[117]:


customer_df = SP_Customer[['Unique_member_identifier','gender','Age','Urban','FSA','City','LanguagePreference','TuesdayAttendee_tendancy','TuesdayAttendee_value','AttendsWithChild_tendancy','AttendsWithChild_value','OnlineTicketPurchaser_tendancy','OnlineTicketPurchaser_value','OpensEmail_tendancy','OpensEmail_value','ClicksEmail_tendancy','ConcessionPurchaser_value']]


# In[118]:


customer_df = customer_df.merge(cust_spend, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(avg_time, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(avghour, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(timeofday, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(carrot_timeplay, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(carrot_total, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(emailorsms, on="Unique_member_identifier", how="left", validate="1:1")
customer_df = customer_df.merge(last_trans_vs_earliest_date, on="Unique_member_identifier", how="left", validate="1:1")


# In[119]:


customer_df.info()


# In[120]:


customer_df['TimeplayPlayer'] = customer_df['TtlTimeplay'].apply(lambda x: 1 if x>0 else 0)
customer_df['CarrotRewardsUser'] = customer_df['points'].apply(lambda x: 1 if x>0 else 0)
customer_df.drop(['TtlTimeplay', 'points'], axis=1, inplace=True)


# In[121]:


customer_df['AvgDaysBetweenMovies'] = customer_df.apply(lambda x: x['longesttime'] if pd.isnull(x['AvgDaysBetweenMovies']) else x['AvgDaysBetweenMovies'], axis=1)
customer_df.drop(['longesttime'], axis=1, inplace=True)


# Questions for Steve:
# * Can we cluster only the users who went to movies?
# * When is it appropriate to use one hot encoding vs hamming

# In[122]:


movie_customers = customer_df.dropna(subset=['TtlNetPoints'])


# In[123]:


movie_customers.info()


# In[124]:


movie_customers["TuesdayAttendee_tendancy"] = movie_customers["TuesdayAttendee_tendancy"].astype(int)
movie_customers["AttendsWithChild_tendancy"] = movie_customers["AttendsWithChild_tendancy"].astype(int)
movie_customers["OnlineTicketPurchaser_tendancy"] = movie_customers["OnlineTicketPurchaser_tendancy"].astype(int)
movie_customers["OpensEmail_tendancy"] = movie_customers["OpensEmail_tendancy"].astype(int)
movie_customers["ClicksEmail_tendancy"] = movie_customers["ClicksEmail_tendancy"].astype(int)


# In[125]:


movie_customers.info()


# In[126]:


#movie_customers.to_csv("customers.csv", index_label=False)


# In[127]:


#combined_movies.to_csv("customer_transactions.csv", index_label=False)


# # Clustering

# In[128]:


import numpy as np

import seaborn as sns

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.cluster import KMeans

from scipy import stats

import scipy.cluster


# ## Encoding

# In[131]:


customers = movie_customers.copy()
X = movie_customers.copy()


# In[132]:


customers['NonTuesday_value'] = 100-customers['TuesdayAttendee_value'] 
customers['NonChild_value'] = 100-customers['AttendsWithChild_value'] 
customers['PercentSpendLess30'] = 100-(100*customers['PercentSpendOver30'])
customers['NonConcession_value'] = 100-customers['ConcessionPurchaser_value']
customers['TechScore'] = 0.2*customers['UserReachableViaSMSEmail']+0.60*customers['OnlineTicketPurchaser_value']+0.1*customers['TimeplayPlayer']+0.05*customers['CarrotRewardsUser']
customers["Group"] = 0.66*customers['NonChild_value'] + 0.17*customers['PercentSpendLess30'] + 0.17*customers['NonConcession_value']
X = customers.drop(columns=['Unique_member_identifier','FSA', 'City', 'LanguagePreference'], axis=1)


# In[133]:


# We pair down to the most relevant features
feature_names = ['AvgSpendPerMovie', 'AvgTimeOfDay', 'Group',
#            'AvgMoviesPerMonth','TechScore', 'Age']
#feature_names = ['AvgSpendPerMovie', 'ModeTime', 'Group',
            'AvgMoviesPerMonth','TechScore', 'Age']


# In[134]:


X = X[feature_names]
numeric_features = X.select_dtypes('number').columns
categorical_features = X.select_dtypes('object').columns
# Fill n/a with 0
X[numeric_features] = X[numeric_features].fillna(0)
#One Hot Encoding
X = pd.get_dummies(X, columns=categorical_features)
col_names = X.columns


# In[135]:


X_scaled = X.copy()
features = X_scaled[numeric_features]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)
X_scaled[numeric_features] = features


# ## Agglomerative Clustering

# In[136]:


aggl = scipy.cluster.hierarchy.linkage(X_scaled, method='ward', metric='euclidean')


# In[137]:


labels = scipy.cluster.hierarchy.fcluster(aggl, 8, criterion="maxclust")


# In[139]:


# Reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

import matplotlib.cm as cm

# Create a subplot with 1 row and 2 columns
fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.6, 0.8])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X_scaled) + (8+1) * 10])

# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
clusterer = aggl
cluster_labels = labels

# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print("For n_clusters =", 8,
      "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

y_lower = 10
for i in range(1,9):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values =         sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 8)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % 8),
             fontsize=14, fontweight='bold')

plt.show()


# ### Cluster Details

# In[155]:


feature_df = pd.DataFrame(X, columns=col_names)
feature_df['cluster'] = pd.Series(cluster_labels, index=feature_df.index)
cluster_means = feature_df.groupby(['cluster']).mean().reset_index()
cluster_means


# In[156]:


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
uns_clust_mean = pd.melt(cluster_means, id_vars=['cluster'], value_vars=feature_names)


# In[157]:


demographics = ['Age']
entertainment = ['AvgMoviesPerMonth']
group=['Group']
tech = ['TechScore']
time = ['AvgTimeOfDay', 'ModeTime']
spending = ['NonTuesday_value','AvgSpendPerMovie']


# In[158]:


uns_clust_mean['Category'] = ""
uns_clust_mean.loc[uns_clust_mean['variable'].isin(spending),'Category'] = "Spending"
uns_clust_mean.loc[uns_clust_mean['variable'].isin(entertainment),'Category'] = "Entertainment"
uns_clust_mean.loc[uns_clust_mean['variable'].isin(demographics),'Category'] = "Demographics"
uns_clust_mean.loc[uns_clust_mean['variable'].isin(group),'Category'] = "Group Habits"
uns_clust_mean.loc[uns_clust_mean['variable'].isin(tech),'Category'] = "Tech Savviness"
uns_clust_mean.loc[uns_clust_mean['variable'].isin(time),'Category'] = "Time of Day"


# In[159]:


def get_percentile(row):
    percentile = stats.percentileofscore(X[row['variable']], row['value'])
    return percentile


# In[160]:


uns_clust_mean['Percentile'] = uns_clust_mean.apply(get_percentile, axis=1)


# ## Clustering Output - Percentiles

# In[161]:


cluster_pt = pd.pivot_table(uns_clust_mean, values='Percentile', index=['Category', 'variable'],
                    columns=['cluster'], aggfunc=np.sum)
cluster_pt


# ## Clustering Output - Actual Values

# In[162]:


cluster_pt_val = pd.pivot_table(uns_clust_mean, values='value', index=['Category', 'variable'],
                    columns=['cluster'], aggfunc=np.sum)
cluster_pt_val


# ## Clustering Output - Scaled Values

# In[163]:


feature_df_scaled = pd.DataFrame(X_scaled, columns=col_names)
feature_df_scaled['cluster'] = pd.Series(cluster_labels, index=customers.index)
cluster_means_scaled = feature_df_scaled.groupby(['cluster']).mean().reset_index()


# In[164]:


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
uns_clust_mean_scaled = pd.melt(cluster_means_scaled, id_vars=['cluster'], value_vars=feature_names)
uns_clust_mean_scaled['Category'] = ""
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(spending),'Category'] = "Spending"
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(entertainment),'Category'] = "Entertainment"
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(demographics),'Category'] = "Demographics"
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(group),'Category'] = "Group Habits"
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(tech),'Category'] = "Tech Savviness"
uns_clust_mean_scaled.loc[uns_clust_mean_scaled['variable'].isin(time),'Category'] = "Time of Day"
uns_clust_mean_scaled['value'] = 100*uns_clust_mean_scaled['value']


# In[165]:


cluster_pt_val_scale = pd.pivot_table(uns_clust_mean_scaled, values='value', index=['Category', 'variable'],
                    columns=['cluster'], aggfunc=np.sum)
cluster_pt_val_scale 


# ## Output Clusters and their Means

# In[166]:


new_cust = movie_customers.copy()
new_cust['cluster'] = pd.Series(cluster_labels, index=customers.index)

di = {1: "Savvy Savers", 2: "Newly Liberated", 3: "Cultural Explorer", 4: "Shapers & Escapers", 5: "Experience Chasers",
      6: "Freedom Seekers", 7: "Discerning Doers", 8: "Adventure Hunters" }

uns_clust_mean['cluster_name'] = uns_clust_mean['cluster'].map(di)

new_cust['cluster_name'] = new_cust['cluster'].map(di)

uns_clust_mean.head()


# ## Output Customers with Attributes

# In[167]:


new_cust.reset_index().drop('index',axis=1).to_csv("customers_clusters.csv", index=False)


# In[168]:


customer_transactions = combined_movies.copy()


# In[169]:


customer_transactions_clusters = customer_transactions.merge(new_cust[['Unique_member_identifier','cluster_name']], how='left')


# In[170]:


#customer_transactions_clusters.to_csv("customer_transactions_clusters.csv", index=False)


# ## Output Scaled Customer Habits

# In[171]:


X_scaled_clusters = X_scaled.copy()


# In[172]:


X_scaled_clusters['cluster'] = pd.Series(cluster_labels, index=customers.index)
X_scaled_clusters['cluster_name'] = X_scaled_clusters['cluster'].map(di)


# In[173]:


#X_scaled_clusters.to_csv("features_scaled.csv", index=False)


# ## Output Unscaled Customer Habits

# In[174]:


X_unscaled_clusters = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=col_names)


# In[175]:


X_unscaled_clusters['cluster'] = pd.Series(cluster_labels)
X_unscaled_clusters['cluster_name'] = X_unscaled_clusters['cluster'].map(di)
#X_unscaled_clusters.to_csv("features_unscaled.csv", index=False)

