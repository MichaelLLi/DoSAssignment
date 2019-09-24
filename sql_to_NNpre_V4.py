import pandas as pd
import os
import pdb
from helpers import master_lists
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np


dir = 'F:/Large Data Work/Lineage'
stop = stopwords.words('english')
trans=pd.read_csv(dir + "/Translation_New.csv", names=['orig', 'freq', 'trans'], usecols=[0, 1, 2], dtype=str)
trans=trans.dropna()
snowball_stemmer = SnowballStemmer("english")

data_interest=pd.read_csv(dir + "/every_pbc_2_15_big.csv", usecols=['facilityid', 'cust_code', 'item_desc', 'product_group', 'latest_date', 'date_in'],dtype={'cust_code': 'str', 'product_group': 'str'})
#data = pd.read_csv('/Users/Daniel/Dropbox/Lineage/sybil_data/sunnyvale_full.csv')

# data.columns=['facilityid','phy_key','product_key','pallet_id','cust_code','cust_name','item_code','lot_key','b_date','julian_date','code_date','location_per_mast','qty_per_mast','status_per_mast','date_in_scan','date_in','orig_qty','location_in','batch_in','earliest_date_scan','earliest_date','latest_date_scan','latest_date','pick_count','qty_picked','location_out','batch_out','ti','hi','case_height','room_type','pallet_height','item_desc','product_group','pickzone1','pickzone2','price_code','location_in_height','location_out_height','location_mast_height','stackable','location_any_height','pallet_weight','location_in_depth','location_out_depth','location_mast_depth','location_any_depth','blast_hours','room_in','room_out','room_mast','room_any','days_duration','weekly_periods','bi_weekly_periods','monthly_periods','time_to_beg_refdate','time_to_refdate','pct_in_period']
#data.columns=['facilityid','phy_key','product_key','pallet_id','cust_code','cust_name','item_code','lot_key','b_date','julian_date','code_date','location_per_mast','qty_per_mast','status_per_mast','date_in_scan','date_in','orig_qty','location_in','batch_in','earliest_date_scan','earliest_date','latest_date_scan','latest_date','pick_count','qty_picked','location_out','batch_out','ti','hi','case_height','room_type','pallet_height','item_desc','product_group','pickzone1','pickzone2','price_code','location_in_height','location_out_height','location_mast_height','stackable','location_any_height','pallet_weight','location_in_depth','location_out_depth','location_mast_depth','location_any_depth','blast_hours','room_in','room_out','room_mast','room_any','days_duration','weekly_periods','bi_weekly_periods','monthly_periods','time_to_beg_refdate','time_to_refdate','pct_in_period','case_length', 'case_width', 'case_height']
#data_interest=data[['facilityid','product_key','pallet_id','cust_code','cust_name','lot_key','item_code','item_desc','product_group','latest_date','date_in','price_code', 'location_in']]


data_interest=data_interest.dropna()


# Time data cleaning
data_interest['date_in']=pd.to_datetime(data_interest['date_in'])
data_interest=data_interest[data_interest.date_in>datetime.strptime('2016-06-30',"%Y-%m-%d")]
data_interest=data_interest[data_interest.date_in<datetime.strptime('2017-06-30',"%Y-%m-%d")]
data_interest['latest_date'] = pd.to_datetime(data_interest['latest_date'])
data_interest=data_interest[data_interest.latest_date<datetime.strptime('2017-06-30',"%Y-%m-%d")]          
                            
                            
data_interest['day_of_week'] = data_interest['date_in'].dt.weekday

data_interest['week_of_year'] = data_interest['date_in'].dt.week

data_interest['duration']=np.ceil((data_interest['latest_date']-data_interest['date_in']).dt.total_seconds() / 60 / 60 / 24).astype(int)
data_interest = data_interest[data_interest['duration'] > 0]
data_interest = data_interest.drop(labels=['date_in', 'latest_date'], axis=1)



# Other data cleaning
data_interest['facilityid']=pd.Categorical(data_interest['facilityid'], categories=master_lists.master_MRSfacilityid_list())

data_interest['product_group'] = pd.Categorical(data_interest['product_group'].str.strip().values, categories=master_lists.master_prodgroup_list())
data_interest['cust_code'] = pd.Categorical(data_interest['cust_code'].str.strip().values, categories=master_lists.master_custcode_list())

#data_interest['cust_name']=data_interest['cust_name'].str.strip().astype('category')  # Changed from string to category

data_interest1=data_interest.dropna()


# Data Investigation Section

'''w1 = data_interest1.groupby(['cust_name', 'product_group', 'item_code', 'day_of_week', 'week_of_year'])
wx = (w1['duration'].agg([np.mean, np.var, lambda x: tuple(np.round(np.cumsum(np.histogram(np.clip(x, 1, 100), bins=list(range(1, 101)), normed=1)[0]), 3))]).rename(columns={'mean': 'avg', '<lambda>': 'cumsum'}))
wx = wx.reset_index()

#data_interest2 = pd.merge(data_interest1, wx, how='inner')'''

data_interest2a = data_interest1.groupby(['facilityid', 'cust_code','product_group', 'item_desc', 'day_of_week', 'week_of_year'])
data_interest2b = data_interest2a['duration'].describe(percentiles=np.arange(.05, 1, .05))
data_interest2b.columns = ['count', 'mean', 'std', 'min', 'perc05', 'perc10', 'perc15', 'perc20', 'perc25','perc30', 'perc35', 'perc40','perc45','perc50', 'perc55','perc60', 'perc65','perc70', 'perc75','perc80', 'perc85','perc90', 'perc95', 'max']
data_interest2b = data_interest2b[data_interest2b['count'] > 10]
data_interest2b = data_interest2b.drop(labels=['count', 'mean', 'std', 'min', 'max'], axis=1)

data_interest2c = data_interest2b.reset_index()
data_interest2d = pd.concat([data_interest2c, pd.get_dummies(data_interest2c['facilityid']), pd.get_dummies(data_interest2c['cust_code']), pd.get_dummies(data_interest2c['product_group'])], axis=1)
data_interest2d = data_interest2d.drop(labels=['product_group', 'cust_code', 'facilityid'], axis=1)



data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace('([A-Za-z/#]+[0-9#/]+[^ ]*|[0-9#/]+[A-Za-z/#]+[^ ]*)', ' ')
data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace('[\s]+', ' ')
data_interest2d['item_desc']=data_interest2d['item_desc'].str.strip()
data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace('[^a-zA-Z ]', ' ')
data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace('[\s]+', ' ')
data_interest2d['item_desc']=data_interest2d['item_desc'].str.strip()

## Take out all the stopwords from the nltk dictionary of stopwords
#for word in stop:
#	data_interest2d['item_desc'] = data_interest2d['item_desc'].str.replace(word, "")

## Re-strip it
#data_interest2d['item_desc']=data_interest2d['item_desc'].str.strip()

# Implement custom translations from Translation_Complete.csv
#data_interest2d['item_desc']=data_interest2d['item_desc'].apply(lambda x: filter(None, x.split(" ")))
#data_interest2d['item_desc']=data_interest2d['item_desc'].apply(lambda x: [snowball_stemmer.stem(y) for y in x])
#data_interest2d['item_desc']=data_interest2d['item_desc'].apply(lambda x: " ".join(x))
data_interest2d['item_desc']=data_interest2d['item_desc'].str.lower()
origin=trans['orig'].values
target=trans['trans'].values
for i in range(trans.shape[0]):
    data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace(" "+str(origin[i])+" ", " "+str(target[i])+" ",regex=True)
for i in range(trans.shape[0]):
    data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace(" "+str(origin[i])+"$", " "+str(target[i]),regex=True)
for i in range(trans.shape[0]):
    data_interest2d['item_desc']=data_interest2d['item_desc'].str.replace("^"+str(origin[i])+" ", str(target[i])+" ",regex=True)




data_interest2d.to_csv("F:/Large Data Work/Lineage/NN_input_V10_big_train_near.csv", index=False)
#data_interest2 = pd.merge(data_interest1, data_interest2c, how='inner')
#data_interest2 = data_interest2.drop(labels='duration', axis=1)

#data_interest3 = pd.concat([data_interest2, pd.get_dummies(data_interest2['cust_name']), pd.get_dummies(data_interest2['product_group'])], axis=1)
#data_interest3 = pd.concat([data_interest2, pd.get_dummies(data_interest2['facilityid']), pd.get_dummies(data_interest2['cust_code']), pd.get_dummies(data_interest2['product_group'])], axis=1)
#data_interest3 = data_interest3.drop(labels=['product_group', 'cust_code', 'facilityid'], axis=1)
#pdb.set_trace()


# Save the datas
#savefull = True
#if savefull:
#    data_interest3.to_csv(dir + "/data/geneva_data_cleaned.csv")
#
#
#np.random.seed(24)
#randomrows=np.random.randint(0, len(data_interest3), 100)
#
#savesample = True
#if savesample:
#    data_interest3.iloc[randomrows].to_csv(dir + "/data/geneva_data_cleaned_sample.csv")


#pdb.set_trace()