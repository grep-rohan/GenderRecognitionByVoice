from math import floor

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read(filename='voice.csv'):
    """
    Read data from file.
    
    :param filename:  Name of file containing data.
    :return: data. (pandas DataFrame)
    """
    data = None
    try:
        data = pd.read_csv(filename)  # read data from csv file
        print('Reading data')
        data['label'] = LabelEncoder().fit_transform(data['label'])  # encode class label
        data = data.sample(frac=1).reset_index(drop=True)  # randomize order
    except FileNotFoundError:
        print('File not found!')

    return data


def scale(data):
    """
    Scale the data between -1 and 1.

    :param data: The data to be scaled. Data type : Pandas DataFrame
    :return: Scaled data. Data Type : Pandas DataFrame
    """
    return (data - data.mean()) / (data.max() - data.min())


def split(data, train_percent=.6):
    """
    Split data into training and testing data.

    :param data: The data to be split.
    :param train_percent: Percentage of data which is training. def = 0.6
    :return: Tuple containing training and testing data.
    """
    train_data = data.iloc[0:floor(train_percent * len(data))]
    test_data = data.iloc[floor(train_percent * len(data)):]

    return train_data, test_data


def visualize(data, style='ggplot', graph_type='line'):
    """
    Visualize data.

    :param data: Data to visualize. (pandas dataframe)
    :param style: matplotlib style. def = 'ggplot'
    :param graph_type: Graph type ('line' or 'area'). def = 'line'
    :return: None
    """
    try:
        plt.style.use(style)
    except OSError:
        print('\nInvalid style!\nUsing ggplot\n')
        plt.style.use('ggplot')
    if graph_type == 'line':
        data.plot()
    elif graph_type == 'area':
        data.plot.area(stacked=False)
    else:
        print('\nInvalid type!\nUsing line')
        data.plot()
    plt.show()
