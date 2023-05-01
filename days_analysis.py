import json
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.fft import fft
import matplotlib.pyplot as plt
import httpx
import asyncio
from datetime import datetime


def plotGraph(json_data,faultName):
    date_range1 = pd.date_range(start=json_data["range1"]["start"], end=json_data["range1"]["end"], inclusive='right')
    magnitute_range1 = [0]*len(date_range1)
    date_range2 = pd.date_range(start=json_data["range2"]["start"], end=json_data["range2"]["end"], inclusive='right')
    magnitute_range2 = [0]*len(date_range2)
    date_range3 = pd.date_range(start=json_data["range3"]["start"], end=json_data["range3"]["end"], inclusive='right')
    magnitute_range3 = [0]*len(date_range3)
    set1 = json_data["to5Years"]
    set2 = json_data["to10Years"]
    set3 = json_data["to15Years"]

    day_data_1,mag_data_1 = timeSeriesData(date_range1,magnitute_range1,set1)
    day_data_2,mag_data_2 = timeSeriesData(date_range2,magnitute_range2,set2)
    day_data_3,mag_data_3 = timeSeriesData(date_range3,magnitute_range3,set3)
    plt.plot(day_data_1,mag_data_1,"r")
    plt.plot(day_data_2,mag_data_2,"g")
    plt.plot(day_data_3,mag_data_3,"b")

    plt.grid(True)
    plt.legend(('today - 5 years','5 years - 10 years','10 years - 15 years'),fontsize=10)
    plt.title('Time series graph for fault line :'+str(faultName),fontsize=12)
    plt.ylabel('Magniitude',fontsize=12)
    plt.xlabel('Date',fontsize=12)
    plt.show()
    plotFourierTransform(day_data_1,mag_data_1,day_data_2,mag_data_2,day_data_3,mag_data_3,faultName)

async def getFaultName(id):
    # url = "http://localhost:3030/earthquake/faultline/"+str(id)
    url = "http://localhost:3030/earthquake/faultline/name/"+str(id)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code == 200:
        name = response.json()
        return name
    else:
        print(f"Request failed with status code {response.status_code}")

async def getFaultData(id):
    # url = "http://localhost:3030/earthquake/faultline/"+str(id)
    url = "http://localhost:3030/earthquake/separate/faultline/"+str(id)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Request failed with status code {response.status_code}")


def timeSeriesData(date_range,magnitute_range,setData):
    date_range_data = date_range
    magnitute_range_data = magnitute_range

    datetime_data = pd.to_datetime([entry["date_utc"] for entry in setData])
    normalized_datetime_data = pd.to_datetime([entry["date_utc"][0:10] for entry in setData])
    magnitude_data = np.array([entry["magnitute"] for entry in setData])
    unique_date = np.unique(normalized_datetime_data)
    new_norm_magnitute = []
    for date in unique_date:
        norm_magnitute_equal = []
        for i, checkDate in enumerate(normalized_datetime_data):
            if checkDate == date:
                norm_magnitute_equal.append(magnitude_data[i])
        new_norm_magnitute.append(max(norm_magnitute_equal))

    for idx_date,each_date in enumerate(date_range):
        for idx_fault_date,fault_date in enumerate(unique_date):
            if each_date == fault_date:
                magnitute_range_data[idx_date]=new_norm_magnitute[idx_fault_date]

    return date_range_data,magnitute_range_data


def plotFourierTransform(x1,y1,x2,y2,x3,y3,faultName):
    n1=len(y1)
    n2=len(y2)
    n3=len(y3)
    N=max(n1,n2,n3)
    Y1 = fft(y1)
    T = 1/N
    X1 = 1/(N*T)*np.arange(N//2)

    amp_c1 =T*np.abs(Y1[0:N//2])

    Y2 = fft(y2)
    X2 = 1/(N*T)*np.arange(N//2)

    amp_c2 =T * np.abs(Y2[0:N//2])

    Y3 = fft(y3)
    X3 = 1/(N*T)*np.arange(N//2)

    amp_c3 =T * np.abs(Y3[0:N//2])

    plt.axes(frameon=True)
    plt.xticks(np.arange(0, N, 100))
    # plt.plot(X1, amp_c1,'r')
    # plt.plot(X2, amp_c2,'g')
    # plt.plot(X3, amp_c3,'b')
    plt.semilogy(X1, amp_c1,'r')
    plt.semilogy(X2, amp_c2,'g')
    plt.semilogy(X3, amp_c3,'b')
    plt.grid(True)
    plt.legend(('today - 5 years','5 years - 10 years','10 years - 15 years'),fontsize=10)
    plt.title('FFT spectrum for fault line :'+str(faultName),fontsize=12)
    plt.ylabel('Amplitude',fontsize=12)
    plt.xlabel('Frequency',fontsize=12)
    plt.show()

    plt.axes(frameon=True)
    plt.xticks(np.arange(0, 300, 100))
    plt.xlim(-10,160)
    # plt.plot(X1, amp_c1,'r')
    # plt.plot(X2, amp_c2,'g')
    # plt.plot(X3, amp_c3,'b')
    plt.semilogy(X1, amp_c1,'r')
    plt.semilogy(X2, amp_c2,'g')
    plt.semilogy(X3, amp_c3,'b')
    plt.grid(True)
    plt.legend(('today - 5 years','5 years - 10 years','10 years - 15 years'),fontsize=10)
    plt.title('FFT spectrum for fault line :'+str(faultName),fontsize=12)
    plt.ylabel('Amplitude',fontsize=12)
    plt.xlabel('Frequency',fontsize=12)

    plt.show()


async def main():
    fault_id = 7
    json_data = await getFaultData(fault_id)
    name_fault = await getFaultName(fault_id)
    fault_name = name_fault["fault_name"]
    plotGraph(json_data,fault_name)
    
 

asyncio.run(main())

