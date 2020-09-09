# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:39:30 2018
Obj: processing geotags on tweets to locate tweets to states; handle geotags at different geographies: CITY, ADMIN, POI, NEIGHBORHOOD, and COUNTRY 
@author: Zhaoya Gong
"""
# 0:tw_id, user_id, timestamp, country_code, place_full_name, place_name, place_type, place_bounding_box, tw_coor, quoted_status_user_id, in_reply_to_user_id, retweeted_user_id, mentioned_user_ids, tt]



import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, box

### Set the boundary for AK state
AKw_bd = (-179.148909, 51.214183, -129.979511, 71.365162)
AKe_bd = (172.46166699999998, 51.357687999999996, 179.77847, 53.010749999999994)



def makeBox(bd):
    return box(bd[0], bd[1], bd[2], bd[3])


def checkContains(loc_str, states):
    coor = eval(loc_str)
    if len(coor) < 4:
        loc = Point(coor)
    else:
        loc = Polygon(coor)
    st = [i for i, s in enumerate(states['geometry']) if s.contains(loc) or s.contains(loc.centroid)]
    if len(st) == 1:
        return states['STUSPS'].loc[st[0]]
    else:
        st = [i for i, s in enumerate(states['geometry'].convex_hull) if s.contains(loc.centroid)]
        if len(st) > 0:
            if len(st) > 1:
                st[0] = states.loc[st, 'geometry'].distance(loc.centroid).idxmin()
            st0 = states['STUSPS'].loc[st[0]]
            if st0 == 'AK':
                if makeBox(AKw_bd).contains(loc.centroid) or makeBox(AKe_bd).contains(loc.centroid):
                    return st0
                else:
                    return np.nan

            return st0
        else:
            return np.nan


def getBounds(stategeo):
    w_bnd = [0., 90., -180., 0.]
    e_bnd = [180., 90., 0., 0.]

    for i in stategeo.geoms:
        bd = i.bounds
        if bd[0] < 0:
            if bd[0] < w_bnd[0]:
                w_bnd[0] = bd[0]
            if bd[2] > w_bnd[2]:
                w_bnd[2] = bd[2]
            if bd[1] < w_bnd[1]:
                w_bnd[1] = bd[1]
            if bd[3] > w_bnd[3]:
                w_bnd[3] = bd[3]
        elif bd[0] > 0:
            if bd[0] < e_bnd[0]:
                e_bnd[0] = bd[0]
            if bd[2] > e_bnd[2]:
                e_bnd[2] = bd[2]
            if bd[1] < e_bnd[1]:
                e_bnd[1] = bd[1]
            if bd[3] > e_bnd[3]:
                e_bnd[3] = bd[3]
    return (w_bnd, e_bnd)




### Set list column names
vals=['tw_id', 'user_id', 'country_code', 'place_full_name', 'place_name', 'place_type', 'place_bounding_box', 'tw_coor',
      'quoted_status_user_id', 'in_reply_to_user_id', 'retweeted_user_id', 'mentioned_user_ids', 'tt']
columns = dict(zip(range(14), vals))
print(columns)

# tweets = pd.read_csv('clean_USelection.txt', delimiter='\t', header=None, skipinitialspace=True, quoting=3)
us_bnd = gpd.read_file(r'cb_2016_us_state_500k/cb_2016_us_state_500k.shp')
df_st = pd.read_csv('states.txt', sep='\t')
st_dt = dict(zip(df_st['state'], df_st['st']))
st_dt['Virgin Islands'] = 'VI'
st_dt['Puerto Rico'] = 'PR'
st_dt['American Samoa'] = 'AS'
st_dt['Guam'] = 'GU'


################################################################################
### Read in data
tweets = pd.read_csv('/Users/gongzz/Workspace/mark20171023.txt', delimiter='\t', header=None, skipinitialspace=True, quoting=3)
mcol = dict(zip(range(3,14), range(2,13)))
mcol[2]=14
tweets = tweets.rename(columns=mcol)

# df_ustrn_vec = df_trn_vec[(df_trn_vec[2]!='Not') & (df_trn_vec[2]!='Null')]
tweets['st']=''
df_ustrn_vec = tweets[(tweets[2]!='Not') & (tweets[2]!='Null')]
train_loc = [row.split(', ') for row in df_ustrn_vec[3]]

print(set(df_ustrn_vec[5]))

#df_ustrn_vec['st']=''

admin = [i.split(', ') for i in set(df_ustrn_vec[df_ustrn_vec[5]=='admin'][3])]
states = set([i[0] for i in admin if len(i)==2])
print(states)


################################################################################
### Handle CITY
city = df_ustrn_vec[df_ustrn_vec[5]=='city'].assign(st=lambda x: x[3].str.split(', '))['st'].apply(
        lambda x: x[1] if (len(x)==2) and (x[1] in st_dt.values()) else np.nan)

#city_l = [i for i in set(city) if i not in st_dt.values()]
#for i in city_l:
#    city[city == i] = df_ustrn_vec.loc[city[city == i].index][6].apply(lambda x: checkContains(x, us_bnd))

df_ustrn_vec.loc[city.index, 'st'] = city

city_bd_flt = df_ustrn_vec.loc[city[city.isna()].index][6].apply(lambda x: checkContains(x, us_bnd))
df_ustrn_vec.loc[city_bd_flt.index, 'st'] = city_bd_flt

print('CITY\n', df_ustrn_vec[(df_ustrn_vec[5]=='city') & ((df_ustrn_vec['st'].isna()) | (df_ustrn_vec['st']==''))])


#st_dt['Kalifornien']='CA'
#st_dt['Califórnia']='CA'
#st_dt['Nova Iorque']='NY'
#st_dt['Nueva York']='NY'
#st_dt['Misuri']='MO'
#st_dt['Distrito de Columbia']='DC'
#st_dt['Virgin Islands']=np.nan


################################################################################
### Handle ADMIN
st_flt = df_ustrn_vec[df_ustrn_vec[5]=='admin'].assign(st=lambda x: x[3].str.split(', '))['st'].apply(
        lambda x: st_dt[x[0]] if (len(x)==2) and (x[0] in st_dt) and (x[1] == 'USA') else np.nan)

df_ustrn_vec.loc[st_flt.index, 'st'] = st_flt

st1_flt = df_ustrn_vec.loc[st_flt[st_flt.isna()].index].assign(st=lambda x: x[3].str.split(' '))['st'].apply(
        lambda x: x[-1] if x[-1] in st_dt.values() else np.nan)

df_ustrn_vec.loc[st1_flt.index, 'st'] = st1_flt

admin_l = df_ustrn_vec[(df_ustrn_vec[5]=='admin') & (df_ustrn_vec['st'].isna())][6].apply(lambda x: checkContains(x, us_bnd))
df_ustrn_vec.loc[(df_ustrn_vec[5]=='admin') & (df_ustrn_vec['st'].isna()), 'st'] = admin_l

print('ADMIN\n', df_ustrn_vec[(df_ustrn_vec[5]=='admin') & ((df_ustrn_vec['st'].isna()) | (df_ustrn_vec['st']==''))])


################################################################################
### Handle POI and NEIGHBORHOOD
neig = df_ustrn_vec[df_ustrn_vec[5]=='neighborhood'][6].apply(lambda x: checkContains(x, us_bnd))
#neig[neig.isna()]
poi = df_ustrn_vec[df_ustrn_vec[5]=='poi'][6].apply(lambda x: checkContains(x, us_bnd))
#poi[poi.isna()]

df_ustrn_vec.loc[poi.index, 'st']=poi
df_ustrn_vec.loc[neig.index, 'st']=neig

print('NEIGHBORHOOD\n', df_ustrn_vec[(df_ustrn_vec[5]=='neighborhood') & ((df_ustrn_vec['st'].isna()) | (df_ustrn_vec['st']==''))])
print('POI\n', df_ustrn_vec[(df_ustrn_vec[5]=='poi') & ((df_ustrn_vec['st'].isna()) | (df_ustrn_vec['st']==''))])


################################################################################
### Handle COUNTRY
country = df_ustrn_vec[df_ustrn_vec[5]=='country'][6].apply(lambda x: checkContains(x, us_bnd))
if len(country[country.isna()]) == len(country):
    print('COUNTRY no state determined!')

df_ustrn_vec = df_ustrn_vec[(df_ustrn_vec[5]!='country')]
# df_ustrn_vec.to_csv('usst6-11.csv', sep='\t', index=False)
