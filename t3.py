import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

number = 365
file = pd.read_csv('file.csv', )

close = file['<Close>'][len(file)-1]
high = max(file['<HIGH>'][-number:])
low = min(file['<LOW>'][-number:])

pp = ((2*close)+high+low)/4
S1 = (2*pp)-high
S2 = pp-(high-low)
S3 = low-2*(high-pp)
R1 = (2*pp)-low
R2 = pp+(high-low)
R3 = high+2*(pp-low)

data = pd.DataFrame()
data['close'] = file['<CLOSE>'][-number:]
data['PP'] = np.array([pp for i in range(number)])
data['S1'] = np.array([S1 for i in range(number)])
data['S2'] = np.array([S2 for i in range(number)])
data['S3'] = np.array([S3 for i in range(number)])
data['R1'] = np.array([R1 for i in range(number)])
data['R2'] = np.array([R2 for i in range(number)])
data['R3'] = np.array([R3 for i in range(number)])

plt.figure(figsize=(12.5, 4.5))
plt.plot(data['close'], label='close', color='blue')
plt.plot(data['PP'], label='PP', color='yellow')
plt.plot(data['S1'], label='S1', color='green')
plt.plot(data['S2'], label='S2', color='green')
plt.plot(data['S3'], label='S3', color='green')
plt.plot(data['R1'], label='R1', color='red')
plt.plot(data['R2'], label='R2', color='red')
plt.plot(data['R3'], label='R3', color='red')
plt.title('Close price of Khesapa')
plt.ylabel('Close price(Rial)')
plt.legend(loc='upper left')
plt.show()