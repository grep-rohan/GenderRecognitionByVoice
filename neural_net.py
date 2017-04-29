"""Preprocess data and train and validate neural net"""
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def read(filename='voice.csv'):
    """
    Read data from file.

    :param filename:  Name of file containing data.
    :return: data. (pandas DataFrame)
    """
    data = None
    try:
        data = pd.read_csv(filename)  # read data from csv file
        print('\nReading data...')
    except FileNotFoundError:
        print('\nFile not found!')

    return data


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


def scale(data):
    """
    Scale the data between -1 and 1.

    :param data: The data to be scaled. Data type : Pandas DataFrame
    :return: Scaled data. Data Type : Pandas DataFrame
    """
    return (data - data.mean()) / (data.max() - data.min())


def run():
    """
    Train neural net and print validation results.
    :return: None
    """
    voice_data = read()  # read data
    print('\nPreprocessing data...')
    x = voice_data.iloc[:, :-1]  # get inputs from data
    x = scale(x)  # scale inputs
    y = voice_data.iloc[:, -1]  # get outputs
    y = LabelEncoder().fit_transform(y)  # encode label
    # split into training and testing data with randomized order
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.6, random_state=1)

    print('\nTraining neural net...')
    neural_net = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='identity', solver='sgd',
                               learning_rate='adaptive', max_iter=2000, verbose=True)
    neural_net.fit(x_train, y_train)  # train neural net

    print('\nSaving trained neural net to file...')
    pickle.dump(neural_net, open('neural_net', 'wb'))

    visualize(pd.Series(neural_net.loss_curve_))  # plot loss curve

    print('\nCalculating Training Accuracy...')
    training = pd.DataFrame(columns=('Actual', 'Predicted'))
    for index in range(len(y_train)):
        predicted_output = neural_net.predict(x_train.iloc[index, :])
        training.loc[index] = [y_train[index], predicted_output[0]]
    correct = 0
    for actual, predicted in zip(training['Actual'], training['Predicted']):
        if actual == predicted:
            correct += 1
    print('Training Accuracy = %.1f%%' % (correct / len(training) * 100))

    print('\nTesting Neural Net...')
    tests = pd.DataFrame(columns=('Actual', 'Predicted'))
    for index in range(len(y_test)):
        predicted_output = neural_net.predict(x_test.iloc[index, :])
        tests.loc[index] = [y_test[index], predicted_output[0]]
    true_pos = true_neg = false_pos = false_neg = 0
    for actual, predicted in zip(tests['Actual'], tests['Predicted']):
        if actual == predicted == 0:
            true_neg += 1
        elif actual == predicted == 1:
            true_pos += 1
        elif actual == 0 != predicted:
            false_neg += 1
        else:
            false_pos += 1
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    print('\nTesting Results:')
    print('Accuracy    = %.1f%%' % (accuracy * 100))
    print('Precision   = %.1f%%' % (precision * 100))
    print('Recall      = %.1f%%' % (recall * 100))
    print('Specificity = %.1f%%' % (specificity * 100))