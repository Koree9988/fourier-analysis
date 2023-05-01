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
    allData = []
    date_range = []
    magnitute_range = []
    day_data=[]
    mag_data=[]
    label_range = []
    for subData in (json_data):
        label_range.append("data:{}-{}".format(subData["range"]["start"],subData["range"]["end"]))
        temp_date_range1 = pd.date_range(start=subData["range"]["start"], end=subData["range"]["end"], inclusive='right')
        temp_magnitute_range = [0]*len(temp_date_range1)
        date_range.append(temp_date_range1)
        magnitute_range.append(temp_magnitute_range)
        allData.append(subData["data"])

    for i in range (len(allData)):
        d_data,m_data= timeSeriesData(date_range[i],magnitute_range[i],allData[i])

        day_data.append(d_data)
        mag_data.append(m_data)
    # plt.plot(day_data_2,mag_data_2,"g")
    # plt.plot(day_data_3,mag_data_3,"b")
    print("{}-{}".format(len(day_data),len(mag_data))) 
    for 
    plt.grid(True)
    plt.legend(label_range,fontsize=10)
    plt.title('Time series graph for fault line :'+str(faultName),fontsize=12)
    plt.ylabel('Magniitude',fontsize=12)
    plt.xlabel('Date',fontsize=12)
    plt.show()
    # plotFourierTransform(day_data_1,mag_data_1,day_data_2,mag_data_2,day_data_3,mag_data_3,faultName)

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
    url = "http://localhost:3030/earthquake/data/separate/faultline/"+str(id)
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
    plt.xticks(np.arange(0, 400, 100))
    plt.xlim(-10,210)
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
    fault_id = 2
    json_data = await getFaultData(fault_id)
    name_fault = await getFaultName(fault_id)
    fault_name = name_fault["fault_name"]
    plotGraph(json_data,fault_name)
    
 

asyncio.run(main())

