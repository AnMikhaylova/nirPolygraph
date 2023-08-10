#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import argparse
import re
import os
from icecream import ic


# #### Функция для конвертации объекта datetime.time в миллисекунды
def totalMiliSec(time):
    return (
            time.microsecond + time.second * 1000000 + time.minute * 60 * 1000000 + time.hour * 60 * 60 * 1000000) / 1000


# #### Функция для фильтрации нулевых стимулов
def filter_stim(s):
    return not re.search(r'C0_\d', s)


# #### Функция для разметки стимулов
def stimPrep(df, sID):  # df = timeline, sID - исследуемый
    stimulRead = filter(filter_stim,
                        list(df.columns[2:]))  # фильтруем список стимулов из датафрейма, чтоб убрать нулевые
    # получаем список нужных стимулов
    stimMilisec = {}
    for s in stimulRead:
        stimMilisec[s] = totalMiliSec(df.loc[df['id'] == sID][s].iloc[0].time())
    return stimMilisec


def stimAndTime(df, t):  # df = timeline in milisec, t - time
    dic = {key: val for key, val in df.items() if ((val <= t) & (t <= val + 10000))}
    return (list(dic.keys())[0].split('_') if dic else [np.nan, np.nan])


# #### Функция для копирования и конвертации фреймов полиграфа
def copyAconvert(df):
    df1 = df.copy(deep=True)
    df1['Time'] = df1['Time'].map(lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
                                             x.hour * 60 * 60 * 1000000) / 1000)
    return df1


# #### Функция для усреднения строк файлов полиграфа
def sortPoly(df, t):
    rows = df[(t <= df['Time']) & (df['Time'] < t + 50)]
    if len(rows) > 1:
        tot = rows['Value'].mean()
    elif len(rows) != 0:
        tot = rows.iloc[0]['Value']
    else:
        tot = 0
    return tot


# #### Функция для усреднения строк файлов FaceReader
def sortEmot(df, t):
    rows = df[(t <= df['Video Time']) & (df['Video Time'] < t + 50)]
    if len(rows) > 1:
        tot = rows.mean().to_frame().T
    elif len(rows) != 0:
        tot = rows.reset_index(drop=True)
    else:
        tot = pd.DataFrame(0, index=[0],
                           columns=['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted',
                                    'Contempt', 'Valence', 'Arousal'])

    return tot


def argParser():
    parser = argparse.ArgumentParser(description='Script polygraph')
    parser.add_argument("--dir", type=str, required=True, help="This is the path to the root data folder")
    # parser.add_argument("--sub", type=str, required=True, help="This is the subject's name")
    args = parser.parse_args()
    dataDir = args.dir + "\\"
    faceData = dataDir + 'facereader\\'
    polyData = dataDir + 'polygraph\\'
    timeData = dataDir + 'time\\'
    polyChannels = ["ABDOMINAL_RESP", "ABS_BLOOD_VOLUME", "BLOOD_VOLUME", "EDA", "HEART_RATE", "PLE", "THORACIC_RESP",
                    "TONIC_EDA", "TREMOR"]

    file_names = os.listdir(faceData)
    subjects = list(map(lambda x: x.replace('.txt', ''), file_names))
    metaInf = {'dataDir': dataDir, 'faceData': faceData, 'polyData': polyData, 'timeData': timeData,
               'polyChannels': polyChannels, 'subjects': subjects}
    return metaInf


# ## Импорт файла timeline
def readTimline(timeData):
    timeline = pd.read_excel(timeData + 'timeline.xlsx')  # парсим время в колонках 2-19
    timeline[timeline.columns[2:20]] = timeline[timeline.columns[2:20]].apply(
        lambda x: pd.to_datetime(x, format="%M:%S:%f", errors='coerce'))
    return timeline


# ## Импорт данных полиграфа
def readPolyDict(polyChannels, polyData, subject):
    polyDict = {}
    readColsP = ['Time', 'Value']

    for p in polyChannels:
        pathP1 = polyData + subject + '\\' + 'Фигуры_' + p + '.txt'  # путь к файлу
        df = pd.read_table(pathP1, sep='\s+', usecols=readColsP)
        df['Value'] = df['Value'].replace(',', '.', regex=True).astype(float)
        df['Time'] = df['Time'].apply(
            lambda x: datetime.datetime.utcfromtimestamp(x // 1000).replace(microsecond=x % 1000 * 1000))
        polyDict[p] = df
    return polyDict


# ## Импорт FaceReader
def readFaceReader(faceData, subject):
    readColsF = ['Video Time', 'Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt',
                 'Valence', 'Arousal']
    colsF = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt', 'Valence', 'Arousal']
    na = ["FIND_FAILED", "FIT_FAILED"]  # значения NA
    pathF1 = faceData + subject + '.txt'
    emotions = pd.read_table(pathF1, sep='\t', skiprows=8, index_col=False, na_values=na, usecols=readColsF)
    emotions[colsF] = emotions[colsF].apply(pd.to_numeric, errors='coerce').fillna(emotions)
    emotions['Video Time'] = pd.to_datetime(emotions['Video Time'], errors='coerce')
    return emotions


# ## Формирование итогового выходного массива
def finalDataFrame(metaInf, subject):
    dataDir = metaInf.get('dataDir')
    faceData = metaInf.get('faceData')
    polyData = metaInf.get('polyData')
    timeData = metaInf.get('timeData')
    polyChannels = metaInf.get('polyChannels')

    timeline = readTimline(timeData)

    polyDict = readPolyDict(polyChannels, polyData, subject)
    emotions = readFaceReader(faceData, subject)

    # #### Скелет итогового датафрейма
    test = pd.DataFrame(columns=['id', 'name', 'time', 'stimul', 'trial'] + list(emotions.columns)[1:] + polyChannels)
    # #### Формирование итогового датафрейма
    # конечное время в миллисекундах, самая ппоследняя временная метка из двух файлов
    if polyDict['EDA']['Time'].iloc[-1].time() > emotions['Video Time'].iloc[-1].time():
        end_time = totalMiliSec(polyDict['EDA']['Time'].iloc[-1].time())
    else:
        end_time = totalMiliSec(emotions['Video Time'].iloc[-1].time())

    subID = timeline[timeline['name'] == subject]['id'].iloc[0]

    df2 = emotions.copy(deep=True)
    df2['Video Time'] = df2['Video Time'].map(
        lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
                   x.hour * 60 * 60 * 1000000) / 1000)

    polyDict2 = {}
    for k in polyDict:
        polyDict2[k] = copyAconvert(polyDict[k])

    # очистка итогового датафрейма
    test.drop(test.index, inplace=True)
    time = 0  # начальное время
    pTotal = {}
    stimMilli = stimPrep(timeline, subID)
    ic()
    while time < int(end_time):
        for k in polyDict2:
            pTotal[k] = sortPoly(polyDict2[k], time)

        eTotal = sortEmot(df2, time)
        stim = stimAndTime(stimMilli, time)
        time += 50  # временной шаг (может регулироваться)
        test.loc[len(test.index)] = [subID, subject, time, stim[0], stim[1], eTotal['Neutral'][0], eTotal['Happy'][0],
                                     eTotal['Sad'][0], eTotal['Angry'][0], eTotal['Surprised'][0],
                                     eTotal['Scared'][0], eTotal['Disgusted'][0], eTotal['Contempt'][0],
                                     eTotal['Valence'][0], eTotal['Arousal'][0], pTotal['ABDOMINAL_RESP'],
                                     pTotal['ABS_BLOOD_VOLUME'], pTotal['BLOOD_VOLUME'], pTotal['EDA'],
                                     pTotal['HEART_RATE'], pTotal['PLE'], pTotal['THORACIC_RESP'], pTotal['TONIC_EDA'],
                                     pTotal['TREMOR']]

        pTotal.clear()
        eTotal = eTotal.iloc[0:0]

    test.to_csv(dataDir + subject + '.csv', encoding='utf-8')
    ic()
    return test


if __name__ == "__main__":
    metaInformation = argParser()
    subjects = metaInformation.get('subjects')
    # для конкретного объекта по его индексу
    finalDataFrame(metaInformation, subjects[2])

    # для нескольких объектов по их индексам
    # for i in 0, 1:
    #     finalDataFrame(metaInformation, subjects[i])

    # для всех участников эксперимента
    # for subj in subjects:
    #     finalDataFrame(metaInformation, subj)