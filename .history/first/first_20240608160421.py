import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
def f(x,theta):
    return np.dot(x,theta)
def standardize(x):
    return (x - np.mean(x)) / np.std(x)
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T
def E(x,y,theta):
    return 0.5 * np.sum((f(x,theta)) - data_y)**2
def regression(x,y,theta):
    count=0
    diff=1
    new_x= to_matrix(standardize(x))
    error = E(new_x,y,theta)
    ETA = 1e-3

    while diff > 1e-4:
        theta = theta - ETA*np.dot(f(new_x,theta)-y,new_x)
        count+=1
        new_error = E(new_x,y,theta)
        diff = abs(new_error-error)
        error = new_error
        print('第{}次,theat={},差值={:.4f}'.format(count,theta,diff))
    return theta
def getdata(path):
    data_frame = pd.read_csv(r'D:\dataenclorse\first\train_dataset.csv')  # skiprows=14
    data_x,data_y = np.array(data_frame['x']), np.array(data_frame['y'])
    return data_x,data_y
def getdata2(path):
    data_frame = pd.read_csv(r'D:\dataenclorse\first\test_dataset.csv')  # skiprows=14
    test_data_x,y= np.array([data_frame['x'], np.array(data_frame['y'])])
    return test_data_x,y
if __name__ == '__main__':
    theta = np.random.randn(3)
    data_x,data_y=getdata('train_dataset.csv')
    new_theta = regression(data_x,data_y,theta)
    x = standardize(data_x)
    plt.scatter(x,data_y,c='blue')
    xx = np.arange(-3,3,0.1)
    plt.plot(xx,f(to_matrix(xx),new_theta),c='red')
    plt.show()
    test_data_x,y=getdata2('test_dataset.csv')
    test_data_y=f(to_matrix(standardize(test_data_x)),new_theta)
    print(test_data_y)

    file_path=r'D:\dataenclorse\first\t.csv'
    with open(file_path,'w',newline='',encoding='utf-8') as f:
        fieldnames=['x','y']
        f_csv = csv.DictWriter(f, fieldnames=fieldnames)
        f_csv.writeheader()
        for i in range(0,len(test_data_y)):
            f_csv.writerow({'x':test_data_x[i],'y':test_data_y[i]})
    pass
