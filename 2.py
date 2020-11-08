# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv('C:\\Users\\JongHyeon\\OneDrive - 서울과학기술대학교\\3학년 2학기\\BA\\팀플\\data\\final.csv', encoding='utf-8')
dt=data.drop(['Unnamed: 0', 'hang_dong', 'apt','transaction_id', 'apartment_id', 'city','jibun','addr_kr','transaction_year_month', 'transaction_date'],axis=1)
head=dt.head(100)
dt.columns

def dummi(dt):
    dum=pd.get_dummies(dt[['dong', 'year_of_completion',
       'floor', 'gu']])
    dum=pd.concat([dum, dt[['exclusive_use_area',
       'transaction_real_price','popularity', 'mail#',
       'femail#', 'korean#', 'korean_mail#', 'korean_femail#',
       'total_foreigner', 'foreigner_mail#', 'foreigner_femail#', 'over65#',
       'square_meters', 'popularity_density', 'households', 'near_subway#',
       'crime5_occur#', 'police#', 'fire_station#', 'total_student#',
       'SNU_stduent#', 'park#', 'park_area', 'care_center#', 'library#',
       'theater#', 'gym#', 'total_store#', 'franchise_stroe#',
       'normal_stroe#']]],axis=1)
    return dum

dt_dum=dummi(dt)
corr=dt_dum.corr()

corr_col=corr.columns
dum_col=dt_dum.columns

complement = list(set(dum_col) - set(corr_col))
