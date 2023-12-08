#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import datetime
import argparse
import re
import os
from icecream import ic
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Окна начала реакции
EDA_WINDOW = {'start': 1200, 'end': 8000}  # КГР
PLE_WINDOW = {'start': 2000, 'end': 9000}  # ФПГ
BREATHE_WINDOW = {'start': 1000, 'end': 90000}  # Дыхание
BLOOD_VOL_WINDOW = {'start': 0, 'end': 10000}  # Давление


# для изменения каналов полиграфа надо изменить значение списка polyChannels
def arg_parser():
    parser = argparse.ArgumentParser(description='Script polygraph')
    parser.add_argument("--dir", type=str, required=True, help="This is the path to the root data folder")
    # parser.add_argument("--sub", type=str, required=True, help="This is the subject's name")
    args = parser.parse_args()
    data_dir = args.dir + "\\"
    face_data = data_dir + 'facereader\\'
    poly_data = data_dir + 'polygraph\\'
    time_data = data_dir + 'time\\'
    # poly_channels = ["ABDOMINAL_RESP", "ABS_BLOOD_VOLUME", "BLOOD_VOLUME", "EDA", "HEART_RATE", "PLE", "THORACIC_RESP",
    #                 "TONIC_EDA", "TREMOR"]
    poly_channels = ["ABDOMINAL_RESP", "BLOOD_VOLUME", "EDA", "PLE", "THORACIC_RESP"]

    file_names = os.listdir(face_data)
    subjects = list(map(lambda x: x.replace('.txt', ''), file_names))  # имена всех участников (субъектов) эксперимента
    meta_inf = {'data_dir': data_dir, 'face_data': face_data, 'poly_data': poly_data, 'time_data': time_data,
                'poly_channels': poly_channels, 'subjects': subjects}
    return meta_inf


# #### Функция для конвертации объекта datetime.time в миллисекунды
def total_mil_sec(time):
    return (
            time.microsecond + time.second * 1000000 + time.minute * 60 * 1000000 +
            time.hour * 60 * 60 * 1000000) / 1000


# #### Функция для фильтрации нулевых стимулов (применяется в функции filter())
def filter_stim(s):
    return not re.search(r'C0_\d', s)


# #### Функция для разметки стимулов (выделение из таблицы временной разметки для конкретного исследуемого)
# Выходное значение: словарь с временной разметкой по стимулам для конкретного исследуемого,
# время представлено в миллисекундах
def stim_prep(df, sID):  # df = timeline, sID - исследуемый
    stimul_read = filter(filter_stim,
                         list(df.columns[2:]))  # фильтруем список стимулов из датафрейма, чтоб убрать нулевые
    # получаем список нужных стимулов
    stim_millisecond = {}
    for s in stimul_read:
        stim_millisecond[s] = total_mil_sec(df.loc[df['id'] == sID][s].iloc[0].time())
    return stim_millisecond


# #### Функция для сопоставления времени в файлах со временем предъявления стимулов
def stim_and_time(timeline, t):  # timeline = timeline in millisecond (dictionary), t - time
    dic = {}
    # dic = {key: val for key, val in df.items() if ((val <= t) & (t <= val + 10000))}
    for i, (key, value) in enumerate(timeline.items()):
        if i < len(timeline) - 1:
            next_key, next_value = list(timeline.items())[i + 1]
            if (value <= t) & (t < next_value):
                dic[key] = value
        else:
            if value <= t:
                dic[key] = value
    if dic:
        dic2 = list(dic.keys())[0].split('_')
        dic2[1] = int(dic2[1])
    else:
        dic2 = [np.nan, np.nan]
    return dic2


# #### Функция для копирования и конвертации фреймов полиграфа
def copy_a_convert(df):
    df1 = df.copy(deep=True)
    df1['Time'] = df1['Time'].map(lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
                                             x.hour * 60 * 60 * 1000000) / 1000)
    return df1


# #### Функция для усреднения строк файлов полиграфа
def sort_poly(df, t):
    rows = df[(t <= df['Time']) & (df['Time'] < t + 50)]
    if len(rows) > 1:
        tot = rows['Value'].mean()
    elif len(rows) != 0:
        tot = rows.iloc[0]['Value']
    else:
        tot = 0
    return tot


# #### Функция для усреднения строк файлов FaceReader
def sort_emot(df, t):
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


# ## Импорт файла timeline
def read_timline(time_data):
    timeline = pd.read_excel(time_data + 'timeline.xlsx')
    timeline[timeline.columns[2:20]] = timeline[timeline.columns[2:20]].apply(  # парсим время в колонках 2-19
        lambda x: pd.to_datetime(x, format="%M:%S:%f", errors='coerce'))
    return timeline


# ## Импорт данных полиграфа
def read_poly_dict(poly_channels, poly_data, subject):
    poly_dict = {}
    read_cols_p = ['Time', 'Value']

    for p in poly_channels:
        pathP1 = poly_data + subject + '\\' + 'Фигуры_' + p + '.txt'  # путь к файлу
        df = pd.read_table(pathP1, sep='\s+', usecols=read_cols_p)
        df['Value'] = df['Value'].replace(',', '.', regex=True).astype(float)
        df['Time'] = df['Time'].apply(
            lambda x: datetime.datetime.utcfromtimestamp(x // 1000).replace(microsecond=x % 1000 * 1000))
        poly_dict[p] = df
    return poly_dict


# ## Импорт FaceReader
def read_face_reader(face_data, subject):
    read_cols_f = ['Video Time', 'Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt',
                   'Valence', 'Arousal']
    cols_f = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Scared', 'Disgusted', 'Contempt', 'Valence', 'Arousal']
    na = ["FIND_FAILED", "FIT_FAILED"]  # значения NA
    path_f1 = face_data + subject + '.txt'
    emotions = pd.read_table(path_f1, sep='\t', skiprows=8, index_col=False, na_values=na, usecols=read_cols_f)
    emotions[cols_f] = emotions[cols_f].apply(pd.to_numeric, errors='coerce').fillna(emotions)
    emotions['Video Time'] = pd.to_datetime(emotions['Video Time'], errors='coerce')
    return emotions


def common_poly_frames(meta_inf, subject):
    poly_data = meta_inf.get('poly_data')
    time_data = meta_inf.get('time_data')
    poly_channels = meta_inf.get('poly_channels')
    timeline = read_timline(time_data)

    poly_dict = read_poly_dict(poly_channels, poly_data, subject)
    poly_frames = {}
    # #### Скелет итогового датафрейма
    for key in poly_dict:
        poly_frames[key] = pd.DataFrame(columns=['id', 'name', 'time', 'stimul', 'trial', key])
        poly_frames[key].drop(poly_frames[key].index, inplace=True)

    poly_dict2 = {}
    for k in poly_dict:
        poly_dict2[k] = copy_a_convert(poly_dict[k])

    end_time = total_mil_sec(poly_dict['EDA']['Time'].iloc[-1].time())
    sub_id = timeline[timeline['name'] == subject]['id'].iloc[0]

    time = 0  # начальное время
    stim_milli = stim_prep(timeline, sub_id)

    ic()
    while time < int(end_time):
        stim = stim_and_time(stim_milli, time)
        for k in poly_dict2:
            p_total = sort_poly(poly_dict2[k], time)
            poly_frames[k].loc[len(poly_frames[k].index)] = [sub_id, subject, time, stim[0], stim[1], p_total]

        time += 50  # временной шаг (может регулироваться)
    ic()
    return poly_frames


# ## Формирование итогового выходного массива
def final_data_frame(meta_inf, subject):
    data_dir = meta_inf.get('data_dir')
    face_data = meta_inf.get('face_data')
    poly_data = meta_inf.get('poly_data')
    time_data = meta_inf.get('time_data')
    poly_channels = meta_inf.get('poly_channels')

    timeline = read_timline(time_data)

    poly_dict = read_poly_dict(poly_channels, poly_data, subject)
    emotions = read_face_reader(face_data, subject)

    # #### Скелет итогового датафрейма
    test = pd.DataFrame(columns=['id', 'name', 'time', 'stimul', 'trial'] + list(emotions.columns)[1:] + poly_channels)

    # #### Формирование итогового датафрейма
    # конечное время в миллисекундах, самая ппоследняя временная метка из двух файлов
    if poly_dict['EDA']['Time'].iloc[-1].time() > emotions['Video Time'].iloc[-1].time():
        end_time = total_mil_sec(poly_dict['EDA']['Time'].iloc[-1].time())
    else:
        end_time = total_mil_sec(emotions['Video Time'].iloc[-1].time())

    sub_id = timeline[timeline['name'] == subject]['id'].iloc[0]

    df2 = emotions.copy(deep=True)
    df2['Video Time'] = df2['Video Time'].map(
        lambda x: (x.microsecond + x.second * 1000000 + x.minute * 60 * 1000000 +
                   x.hour * 60 * 60 * 1000000) / 1000)

    poly_dict2 = {}
    for k in poly_dict:
        poly_dict2[k] = copy_a_convert(poly_dict[k])

    # очистка итогового датафрейма
    test.drop(test.index, inplace=True)
    time = 0  # начальное время
    pTotal = {}
    stim_milli = stim_prep(timeline, sub_id)
    ic()
    while time < int(end_time):
        for k in poly_dict2:
            pTotal[k] = sort_poly(poly_dict2[k], time)

        e_total = sort_emot(df2, time)
        stim = stim_and_time(stim_milli, time)
        time += 50  # временной шаг (может регулироваться)
        test.loc[len(test.index)] = [sub_id, subject, time, stim[0], stim[1], e_total['Neutral'][0],
                                     e_total['Happy'][0],
                                     e_total['Sad'][0], e_total['Angry'][0], e_total['Surprised'][0],
                                     e_total['Scared'][0], e_total['Disgusted'][0], e_total['Contempt'][0],
                                     e_total['Valence'][0], e_total['Arousal'][0], pTotal['ABDOMINAL_RESP'],
                                     pTotal['ABS_BLOOD_VOLUME'], pTotal['BLOOD_VOLUME'], pTotal['EDA'],
                                     pTotal['HEART_RATE'], pTotal['PLE'], pTotal['THORACIC_RESP'], pTotal['TONIC_EDA'],
                                     pTotal['TREMOR']]

        pTotal.clear()
        e_total = e_total.iloc[0:0]

    test.to_csv(data_dir + subject + '.csv', encoding='utf-8', index=False)
    ic()
    return test


def normalize_final_df(final_df):
    # удаляем строки, которые не относятся к предъявлениям
    df = final_df[final_df['trial'].notnull()]
    # массив по трем предъявлениям
    # range(5, df.shape[1]) индексы колонок для нормализации
    for trial_num in 1, 2, 3:
        for col in range(5, df.shape[1]):
            df.loc[df['trial'] == trial_num, df.columns[col]] = zscore(
                df.loc[df['trial'] == trial_num, df.columns[col]])
    return df


def normalize_poly_df(poly_frame):
    # удаляем строки, которые не относятся к предъявлениям
    sorted = poly_frame[poly_frame['trial'].notnull()]
    df = sorted.loc[sorted.iloc[:, -1] != 0]
    # массив по трем предъявлениям
    for trial_num in 1, 2, 3:
        df.loc[df['trial'] == trial_num, df.columns[-1]] = zscore(df.loc[df['trial'] == trial_num, df.columns[-1]])
    return df


def window(start, end, df):
    start_time = df['time'].iloc[0] + start
    end_time = df['time'].iloc[0] + start + end
    sort = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    return sort


def metrics_EDA(df):
    groups = df.groupby(["trial", "stimul"])  # выделяем отдельные датафреймы по предъявлению и стимулу
    result_df = pd.DataFrame(columns=['id', 'name', 'stimul', 'trial', 'EDA'])
    result_df.drop(result_df.index, inplace=True)
    id_num = df.iloc[0]['id']
    name_subj = df.iloc[0]['name']
    for name, group in groups:
        win = window(EDA_WINDOW['start'], EDA_WINDOW['end'], group)
        title = name_subj + " stimul:" + name[1] + " trial:" + str(name[0])
        plt.title(title)
        plt.xlabel("time, ms")
        plt.ylabel("EDA")
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.plot(win['time'], win['EDA'])
        plt.show()
        max_index = win['EDA'].idxmax()
        sliced = win.loc[:max_index, :]
        reverse = sliced.iloc[::-1]
        min_index = reverse['EDA'].idxmin()
        jump_value = win.loc[max_index, 'EDA'] - win.loc[min_index, 'EDA']
        new_row = {'id': id_num, 'name': name_subj, 'stimul': name[1], 'trial': name[0], 'EDA': jump_value}
        result_df.loc[len(result_df.index)] = new_row
    return result_df


def metrics_resp(dataframe):
    groups = dataframe.groupby(["trial", "stimul"])  # выделяем отдельные датафреймы по предъявлению и стимулу
    result_df = pd.DataFrame(columns=['id', 'name', 'stimul', 'trial', 'THORACIC_RESP'])
    result_df.drop(result_df.index, inplace=True)
    id_num = dataframe.iloc[0]['id']
    name_subj = dataframe.iloc[0]['name']
    for name, group in groups:
        df = window(BREATHE_WINDOW['start'], BREATHE_WINDOW['end'], group)
        title = name_subj + " stimul:" + name[1] + " trial:" + str(name[0])
        plt.title(title)
        plt.xlabel("time, ms")
        plt.ylabel("THORACIC_RESP")
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.plot(df['time'], df['THORACIC_RESP'])
        plt.show()
        # вычисляем разности между соседними значениями x и y
        df['dx'] = df['time'].diff()
        df['dy'] = df['THORACIC_RESP'].diff()
        # возводим каждую разность в квадрат
        df['dx^2'] = df['dx'] ** 2
        df['dy^2'] = df['dy'] ** 2
        # суммируем все полученные значения
        sum_dx2_dy2 = df['dx^2'].sum() + df['dy^2'].sum()
        # извлекаем квадратный корень из суммы
        length = sum_dx2_dy2 ** 0.5
        new_row = {'id': id_num, 'name': name_subj, 'stimul': name[1], 'trial': name[0], 'THORACIC_RESP': length}
        result_df.loc[len(result_df.index)] = new_row

    return result_df


def metrics_vol(df):
    groups = df.groupby(["trial", "stimul"])  # выделяем отдельные датафреймы по предъявлению и стимулу
    result_df = pd.DataFrame(columns=['id', 'name', 'stimul', 'trial', 'BLOOD_VOLUME'])
    result_df.drop(result_df.index, inplace=True)
    id_num = df.iloc[0]['id']
    name_subj = df.iloc[0]['name']
    for name, group in groups:
        win = window(BLOOD_VOL_WINDOW['start'], BLOOD_VOL_WINDOW['end'], group)
        title = name_subj + " stimul:" + name[1] + " trial:" + str(name[0])
        plt.title(title)
        plt.xlabel("time, ms")
        plt.ylabel("BLOOD_VOLUME")
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.plot(win['time'], win['BLOOD_VOLUME'])
        plt.show()

        new_row = {'id': id_num, 'name': name_subj, 'stimul': name[1], 'trial': name[0], 'BLOOD_VOLUME': 0}
        result_df.loc[len(result_df.index)] = new_row
    return result_df


def metrics_ple(df):
    groups = df.groupby(["trial", "stimul"])  # выделяем отдельные датафреймы по предъявлению и стимулу
    result_df = pd.DataFrame(columns=['id', 'name', 'stimul', 'trial', 'PLE'])
    result_df.drop(result_df.index, inplace=True)
    id_num = df.iloc[0]['id']
    name_subj = df.iloc[0]['name']
    for name, group in groups:
        win = window(PLE_WINDOW['start'], PLE_WINDOW['end'], group)
        local_max = argrelextrema(win['PLE'].to_numpy(), np.greater)[0]  # индексы элементов с локальным максимумом
        local_min = argrelextrema(win['PLE'].to_numpy(), np.less)[0]  # индексы элементов с локальным минимумом
        threshold = 0.1  # порог для отброса слишком маленьких скачков
        has_min = False
        has_max = False
        min_ind = 0
        max_ind = 0
        diff = []  # массив с величинами скачка между минимумами и максимумами
        for i in range(0, win['PLE'].shape[0]):
            if i in local_min:
                min_ind = i
                has_min = True
            if (i in local_max) & has_min:
                has_max = True
                max_ind = i
            if has_min & has_max:
                d = win['PLE'].iloc[max_ind] - win['PLE'].iloc[min_ind]
                diff.append(d)
                has_min = False
                has_max = False

        has_threshold = any(diff[i] < threshold for i in range(len(diff) - 1))
        save_diff = diff.copy()
        while has_threshold:
            diff.remove(min(diff))
            has_threshold = any(diff[i] < threshold for i in range(len(diff) - 1))
        value = min(diff)
        title = name_subj + " stimul:" + name[1] + " trial:" + str(name[0])
        plt.title(title)
        plt.xlabel("time, ms")
        plt.ylabel("PLE")
        # включаем основную сетку
        plt.grid(which='major')
        # включаем дополнительную сетку
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.plot(win['time'], win['PLE'])
        for i in local_max:
            plt.plot(win['time'].iloc[i], win['PLE'].iloc[i], "x")
        for i in local_min:
            plt.plot(win['time'].iloc[i], win['PLE'].iloc[i], "x")
        plt.show()

        new_row = {'id': id_num, 'name': name_subj, 'stimul': name[1], 'trial': name[0], 'PLE': value}
        result_df.loc[len(result_df.index)] = new_row
    return result_df


if __name__ == "__main__":
    meta_information = arg_parser()
    subjects = meta_information.get('subjects')

    # для конкретного объекта по его индексу

    # frame2 = finalDataFrame(meta_information, subjects[2])
    # frame2Norm = normalizeFinalDF(frame2)
    # t = common_poly_frames(meta_information, subjects[2])
    # norm = normalize_poly_df(t['EDA'])
    # plt1 = plt.plot(norm['time'], norm['EDA'])
    # win = window(EDA_WINDOW['start'], EDA_WINDOW['end'], norm)
    # plt2 = plt.plot(win['time'], win['EDA'])
    # val = metrics_EDA(win)

    poly_data_dict = common_poly_frames(meta_information, subjects[1])
    # метрика для кгр
    # norm_EDA = normalize_poly_df(poly_data_dict['EDA'])
    # val_EDA = metrics_EDA(norm_EDA)
    # print(val_EDA)

    # метрика для дыхания
    # norm_RESP = normalize_poly_df(poly_data_dict['THORACIC_RESP'])
    # val_RESP = metrics_resp(norm_RESP)
    # print(val_RESP)

    # метрика для давления
    # norm_VOL = normalize_poly_df(poly_data_dict['BLOOD_VOLUME'])
    # plt.plot(norm_VOL['time'], norm_VOL['BLOOD_VOLUME'])
    # plt.show()
    # val_VOL = metrics_vol(norm_VOL)

    # метрика для фпг
    norm_PLE = normalize_poly_df(poly_data_dict['PLE'])
    # plt.plot(norm_PLE['time'], norm_PLE['PLE'])
    # plt.show()
    val_PLE = metrics_ple(norm_PLE)

    # для нескольких объектов по их индексам
    # for i in 0, 1, 2:
    #     norm_EDA = normalize_poly_df(common_poly_frames(meta_information, subjects[i])['EDA'])
    #     val_EDA = metrics_EDA(norm_EDA)
    #     arr.append(val_EDA)
    #     norm_RESP = normalize_poly_df(common_poly_frames(meta_information, subjects[i])['THORACIC_RESP'])
    #     val_resp = metrics_resp(norm_RESP)
    #     arr2.append(val_resp)

    # для всех участников эксперимента
    # for subj in subjects:
    #     finalDataFrame(meta_information, subj)
