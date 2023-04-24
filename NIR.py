#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime 
import re 
from icecream import ic 


# ## Задание путей и имени исследуемого 
# 

# In[2]:


dataDir = 'D:\\dataPoly\\'
faceData = dataDir + 'facereader\\'
polyData = dataDir + 'polygraph\\'
timeData = dataDir + 'time\\'
polyChannels = ["ABDOMINAL_RESP", "ABS_BLOOD_VOLUME", "BLOOD_VOLUME", "EDA", "HEART_RATE", "PLE", "THORACIC_RESP", "TONIC_EDA", "TREMOR"]
subject = r'Базаев Кирилл'


# ## Импорт файла timeline

# In[3]:


custom_date_parser = lambda x: datetime.datetime.strptime(x, "%M:%S:%f") #парсер для времени
timeline = pd.read_excel(timeData + 'timeline.xlsx', parse_dates = list(range(2,20)), date_parser = custom_date_parser) #парсим время в колонках 2-19
timeline.info()


# In[4]:


timeline.head()


# ## Импорт данных полиграфа

# In[47]:


polyDict = {}
readColsP = ['Time', 'Value']

for p in polyChannels:
    pathP1 = polyData + subject + '\\' + 'Фигуры_' + p + '.txt' #путь к файлу
    df = pd.read_table(pathP1 ,sep='\s+', usecols = readColsP)
    df['Value'] = df['Value'].replace(',', '.', regex = True).astype(float)
    df['Time'] = df['Time'].apply(lambda x: datetime.datetime.utcfromtimestamp(x//1000).replace(microsecond=x%1000*1000))
    polyDict[p] = df

for p in polyChannels:
    print(polyDict[p])


# In[54]:


#bd_vl['Time'].dt.time


# In[8]:


#emotions['Video Time'] = pd.to_datetime(emotions['Video Time'])
#pd.to_datetime(emotions['Video Time']).dt.time


# ## Импорт FaceReader

# In[37]:


#cols = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared',
#       'Disgusted', 'Contempt', 'Valence', 'Arousal', 'Y - Head Orientation', 
#        'X - Head Orientation','Z - Head Orientation', 'Quality']

readColsF = ['Video Time', 'Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt', 'Valence', 'Arousal'] 
colsF = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt', 'Valence', 'Arousal'] 
#conv = dict.fromkeys(cols, lambda x: pd.to_numeric(x, errors='coerce'))
na = ["FIND_FAILED", "FIT_FAILED"] #значения NA
pathF1 = faceData + subject + '.txt'

emotions = pd.read_table(pathF1, sep='\t', skiprows=8, index_col=False, parse_dates = ['Video Time'], na_values = na, usecols = readColsF)
#converters= conv

emotions[colsF] = emotions[colsF].apply(pd.to_numeric,errors='coerce').fillna(emotions)

#for i in range(26, 68):
#    emotions.iloc[:,i] = pd.to_numeric(emotions.iloc[:,i], errors='coerce').fillna(0, downcast='infer')


# In[38]:


emotions.info()


# In[39]:


emotions.iloc[:10] #проверка значений NaN


# ## Формирование итогового выходного массива

# #### Скелет итогового датафрейма

# In[40]:


subID = timeline[timeline['name'] == subject]['id'][0]
test = pd.DataFrame(columns= ['id', 'name', 'time', 'stimul', 'trial'] + list(emotions.columns)[1:] + polyChannels)
test


# #### Функция для конвертации объекта datetime.time в миллисекунды

# In[41]:


def totalMiliSec(time):
    return (time.microsecond + time.second * 1000000 + time.minute * 60 * 1000000 + time.hour * 60 * 60 * 1000000)/1000


# #### Функция для фильтрации нулевых стимулов

# In[42]:


def filter_stim(s): 
    return not re.search(r'C0_\d', s)


# In[43]:


stimulRead = filter(filter_stim, list(timeline.columns[2:])) #фильтруем список стимулов из датафрейма, чтоб убрать нулевые

for s in stimulRead:
    print(totalMiliSec(timeline.loc[timeline['id'] == subID][s][0].time())) #время предявления каждого (ненулевого) стимула в мс


# #### Функция для усреднения строк файлов полиграфа

# In[53]:


def sortPoly(df, t):
    df1 = df.copy(deep=True)
    df1['Time'] = df1['Time'].map(lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
                                         x.hour * 60 * 60 * 1000000) / 1000)
    rows = df1[(t <= df1['Time']) & (df1['Time'] < t + 50)]
    #print(rows['Value'])
    if len(rows) > 1:
        tot = rows['Value'].mean()
    elif len(rows) != 0:
        tot = rows.iloc[0]['Value']
    else:
        tot = 0
    return tot


# #### Формирование итогового датафрейма

# In[51]:


#конечное время
end_time = 0 #в миллисекундах, самая ппоследняя временная метка из двух файлов
if (polyDict['EDA']['Time'].iloc[-1].time() > emotions['Video Time'].iloc[-1].time()):
    end_time = totalMiliSec(polyDict['EDA']['Time'].iloc[-1].time())
else:
    end_time = totalMiliSec(emotions['Video Time'].iloc[-1].time())

print (end_time)

df2 = emotions.copy(deep=True)
df2['Video Time'] = df2['Video Time'].map(
    lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
               x.hour * 60 * 60 * 1000000) / 1000)


# In[54]:


# очистка итогового датафрейма
ic()  # start
test.drop(test.index, inplace=True)

time = 0  # начальное время
pTotal = {}
while time < int(end_time):
 
    for k in polyDict:
        tot = sortPoly(polyDict[k], time)
        pTotal[k] = tot
    # ic(pTotal)

    eRows = df2[(time <= df2['Video Time']) & (df2['Video Time'] < time + 50)]

    if len(eRows) > 1:
        # ic('if1')
        # ic()
        # ic(eRows)
        # eTotal = pd.DataFrame(np.nanmean(eRows))
        eTotal = eRows.mean().to_frame().T
        # ic(eTotal)
    elif len(eRows) != 0:
        # ic('if2')
        # ic()
        # ic(eRows)
        eTotal = eRows.reset_index(drop=True)
        # ic(eTotal)
    else:
        # ic('if3')
        # ic()
        eTotal = pd.DataFrame(0, index=[0],
                              columns=['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted',
                                       'Contempt', 'Valence', 'Arousal'])

    time += 50  # временной шаг (может регулироваться)
    test.loc[len(test.index)] = [subID, subject, time, 0, 0, eTotal['Neutral'][0], eTotal['Happy'][0],
                                 eTotal['Sad'][0], eTotal['Angry'][0], eTotal['Surprised'][0],
                                 eTotal['Scared'][0], eTotal['Disgusted'][0], eTotal['Contempt'][0],
                                 eTotal['Valence'][0], eTotal['Arousal'][0], pTotal['ABDOMINAL_RESP'], pTotal['ABS_BLOOD_VOLUME'], pTotal['BLOOD_VOLUME'], pTotal['EDA'], pTotal['HEART_RATE'], pTotal['PLE'], pTotal['THORACIC_RESP'], pTotal['TONIC_EDA'], pTotal['TREMOR']]

    # pRows.drop(pRows.index, inplace=True)  # очищаем промежуточные датафреймы
    eRows.drop(eRows.index, inplace=True)
    pTotal.clear()

ic()  # stop
test.to_csv(dataDir + 'file.csv', encoding='utf-8')


# In[ ]:




