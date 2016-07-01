
# coding: utf-8

# In[5]:

FOLDER = 'resources/'

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

#%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rc('font', family='Times New Roman')

pd.set_option('max_rows', 50)

time_format = '%b %d, %H:%M'

start_time = time.time()
current_time = pd.read_csv(FOLDER + 'current_time.csv').current_time[0]
twr          = pd.read_csv(FOLDER + 'team_work_region.csv', converters={'twr':str})
links        = pd.read_csv(FOLDER + 'link.csv')
stations     = pd.read_csv(FOLDER + 'station.csv', converters={'station':str})
train_info   = pd.read_csv(FOLDER + 'train_info.csv', converters={'train': str, 'st_from':str, 'st_to':str, 'oper_location':str,
                                                                 'st_from':str, 'st_to':str})
train_plan   = pd.read_csv(FOLDER + 'slot_train.csv', converters={'train': str, 'st_from':str, 'st_to':str})
loco_info    = pd.read_csv(FOLDER + 'loco_attributes.csv', converters={'train':str, 'loco':str, 'depot':str,
                                                                      'st_from':str, 'st_to':str})
loco_plan    = pd.read_csv(FOLDER + 'slot_loco.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
team_info    = pd.read_csv(FOLDER + 'team_attributes.csv', converters={'team':str,'depot':str, 'oper_location':str,                                                                  'st_from':str, 'st_to':str, 'loco':str, 'depot_st':str})
team_plan    = pd.read_csv(FOLDER + 'slot_team.csv', converters={'team':str,'loco':str, 'st_from':str, 'st_to':str})
loco_series  = pd.read_csv(FOLDER + 'loco_series.csv')

team_info.regions = team_info.regions.apply(literal_eval)
st_names = stations[['station', 'name', 'esr']].drop_duplicates().set_index('station')
print('Planning start time: %s (%d)' % (time.strftime(time_format, time.localtime(current_time)), current_time))


# In[6]:

# Мержим таблицы _plan и _info для поездов, локомотивов и бригад
# Добавляем во все таблицы названия станций на маршруте и времена отправления/прибытия в читабельном формате

def add_info(df):    
    if 'st_from' in df.columns:
        df['st_from_name'] = df.st_from.map(st_names.name)
    if 'st_to' in df.columns:
        df['st_to_name'] = df.st_to.map(st_names.name)
    if 'time_start' in df.columns:
        df['time_start_norm'] = df.time_start.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'time_end' in df.columns:
        df['time_end_norm'] = df.time_end.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'oper_location' in df.columns:
        df['oper_location_name'] = df.oper_location.map(st_names.name)    
        df.oper_location_name.fillna(0, inplace=True)
    if ('oper_location' in df.columns) & ('st_from' in df.columns) & ('st_to' in df.columns):        
        df['loc_name'] = df.oper_location_name
        df.loc[df.loc_name == 0, 'loc_name'] = df.st_from_name + ' - ' + df.st_to_name
    
add_info(train_plan)
add_info(loco_plan)
add_info(team_plan)
add_info(loco_info)
add_info(team_info)
add_info(train_info)
train_plan = train_plan.merge(train_info, on='train', suffixes=('', '_info'), how='left')
loco_plan = loco_plan.merge(loco_info, on='loco', suffixes=('', '_info'), how='left')
team_plan = team_plan.merge(team_info, on='team', suffixes=('', '_info'), how='left')
team_plan['team_type'] = team_plan.team.apply(lambda x: 'Реальная' if str(x)[0] == '2' else 'Фейковая')


# In[7]:

def nice_time(t):
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''


# In[17]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
team_plan['depot_name'] = team_plan.depot.map(st_names.name)
cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols]

team_plan[team_plan.number == 9205004408][cols]


# In[63]:

team_info['depot_name'] = team_info.depot.map(st_names.name)
team_info['in_plan'] = team_info.team.isin(team_plan[team_plan.state == 1].team)
team_info['oper_time_f'] = team_info.oper_time.apply(nice_time)
cols = ['team', 'number', 'depot_name', 'state']
a = team_info[(team_info.ttype == 1) & (team_info.loc_name == 'СЛЮДЯНКА I') 
          #& (team_info.depot_name.isin(['СЛЮДЯНКА I']))
          & (team_info.oper_time < current_time + 24 * 3600)]
a.depot_name.value_counts()


# In[90]:

b = a[a.in_plan == True]
#cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
#team_plan[team_plan.team.isin(b.team)][cols]
cols = ['team', 'regions', 'number', 'depot_name', 'ready_type', 'depot_st', 'depot_time', 'return_st', 'oper_time_f', 'state']
b = a[(a.in_plan) & (a.regions.apply(lambda x: '2002118236' in x))].sort_values('oper_time')[cols]
print(b.team.count()) # всего 30 слюдянковских бригад, которые могут ездить до Иркутска
# Еще 29 бригад из Зимы и Иркутска - они тоже могут ехать в нечетную сторону
team_plan[(team_plan.team.isin(b.team)) & (team_plan.st_from_name == 'СЛЮДЯНКА I') & (team_plan.state.isin([0, 1]))].st_to_name.value_counts()


# In[53]:

twr = pd.read_csv(FOLDER + 'team_work_region.csv')
twr['link'] = twr.link.apply(literal_eval)
twr['st_from_name'] = twr.link.apply(lambda x: x[0]).map(st_names.name)
twr['st_to_name'] = twr.link.apply(lambda x: x[1]).map(st_names.name)
twr[twr.twr == 2002118236]


# In[67]:

train_plan['train_type'] = train_plan.train.apply(lambda x: x[0])
train_plan[(train_plan.st_from_name == 'СЛЮДЯНКА I') & (train_plan.st_to_name == 'СЛЮДЯНКА II')
          & (train_plan.time_start >= current_time) & (train_plan.time_start < current_time + 24 * 3600)].train_type.value_counts()

# Всего 96 поездов в нечетную сторону из Слюдянки!!!
# Из них всего 15 локомотивов резервом и 81 настоящий поезд


# In[ ]:

Итого на Слюдянку надо 96 бригад. А есть только 59 (на начало планирования)
Надо где-то найти еще 37. 
Еще 10 бригад едут от Иркутска в Слюдянку на начало планирования. Осталось 27.


# In[91]:

cols = ['team', 'number', 'depot_name', 'depot_st', 'depot_time', 'state', 'loc_name', 'oper_time_f']
team_info['link'] = list(zip(team_info.st_from, team_info.st_to))
links = pd.read_csv(FOLDER + 'link.csv', dtype={'st_from':str, 'st_to':str})
links['link'] = list(zip(links.st_from, links.st_to))
team_info['dir'] = team_info.link.map(links.set_index('link')['dir'])
team_info[(team_info.depot_name.isin(['ЗИМА', 'ИРКУТСК-СОРТИРОВОЧНЫЙ'])) & (team_info.state.isin(['2','3','4']) == False)
         & (team_info['dir'] == 0)].sort_values('oper_time')[cols]


# In[99]:

cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[team_plan.team == '200200035170'][cols]


# In[88]:

cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[(team_plan.depot_name == st_name) & (team_plan.st_to_name == 'ГОНЧАРОВО')
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 8 * 3600)
         & (team_plan.state == 1) & (team_plan.st_from_name == st_name)].sort_values('time_start')[cols]


# In[105]:

team_info[(team_info.depot_st == '-1') & (team_info.depot_time == -1)].team.count() / team_info.team.count()


# In[109]:

train_info['in_plan'] = train_info.train.isin(train_plan.train)
train_info[train_info.in_plan == False].train.count() / train_info.train.count()


# In[119]:

print(nice_time(current_time))
train_info['oper_time_f'] = train_info.oper_time.apply(nice_time)
train_info[train_info.in_plan == False][['train', 'number', 'ind434', 'joint', 'oper_time_f', 'loc_name']]

