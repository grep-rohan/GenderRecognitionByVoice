import warnings

from pandas import Series, DataFrame
from sklearn.neural_network import MLPClassifier

import preprocess

warnings.filterwarnings("ignore")

attributes = ('meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR')

if __name__ == '__main__':
    data = preprocess.read()
    train, test = preprocess.split(data)
    training_inputs = [[train.iloc[index][attribute] for attribute in attributes] for index in range(len(train))]
    training_outputs = [train.iloc[i]['label'] for i in range(len(train))]
    testing_inputs = [[test.iloc[index][attribute] for attribute in attributes] for index in range(len(test))]
    testing_outputs = [test.iloc[i]['label'] for i in range(len(test))]

    neural_net = MLPClassifier(hidden_layer_sizes=(9, 9), activation='identity', solver='sgd',
                               learning_rate='adaptive', max_iter=2000, verbose=True)
    neural_net.fit(training_inputs, training_outputs)
    preprocess.visualize(Series(neural_net.loss_curve_))

    tests = DataFrame(columns=('Actual', 'Predicted'))
    index = 0
    for testing_input, testing_output in zip(testing_inputs, testing_outputs):
        predicted_output = neural_net.predict(testing_input)
        tests.loc[index] = [testing_output, predicted_output[0]]
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
    try:
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    except ZeroDivisionError:
        pass
    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        pass
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        pass
    try:
        specificity = true_neg / (true_neg + false_pos)
    except ZeroDivisionError:
        pass

    print('\nAccuracy    = %.2f%%' % (accuracy * 100))
    print('Precision   = %.2f%%' % (precision * 100))
    print('Recall      = %.2f%%' % (recall * 100))
    print('Specificity = %.2f%%' % (specificity * 100))
