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
        print('Reading data')
    except FileNotFoundError:
        print('File not found!')

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


if __name__ == '__main__':
    voice_data = read()  # read data
    x = voice_data.iloc[:, :-1]
    x = scale(x)
    y = voice_data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.6, random_state=1)

    neural_net = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd',
                               learning_rate='adaptive', max_iter=2000, verbose=True)
    neural_net.fit(x_train, y_train)
    visualize(pd.Series(neural_net.loss_curve_))

    tests = pd.DataFrame(columns=('Actual', 'Predicted'))
    index = 0
    for index in range(len(y_test)):
        predicted_output = neural_net.predict(x_test.iloc[index, :])
        tests.loc[index] = [y_test[index], predicted_output[0]]
        index += 1

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

    accuracy = precision = recall = specificity = 0
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)

    print('\nAccuracy    = %.2f%%' % (accuracy * 100))
    print('Precision   = %.2f%%' % (precision * 100))
    print('Recall      = %.2f%%' % (recall * 100))
    print('Specificity = %.2f%%' % (specificity * 100))
