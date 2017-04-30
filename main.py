import os
import pickle
from subprocess import run

import pandas as pd

import neural_net
import sound_recorder

if __name__ == '__main__':
    exit_ = False
    while not exit_:
        print('\nMenu')
        print('1. Train Neural Net')
        print('2. Analyse Voice')
        print('3. Exit')
        option = input('Enter Option Number: ')

        if option == '1':
            neural_net.run()
        elif option == '2':
            if not os.path.isfile('trained_neural_net'):  # check if neural_net file exists
                print('\nNeural net not trained. First train the neural net.')
            else:
                sound_recorder.run()

                print('\nExtracting data from recorded voice...\n')
                run(['Rscript', 'getAttributes.r', os.getcwd()])

                print('\nPreprocessing extracted data...')
                data = pd.read_csv('output/voiceDetails.csv')
                del data['peakf'], data['sound.files'], data['selec'], data['duration']
                dataset = pd.read_csv('voice.csv')
                dataset = dataset.iloc[:, :-1]
                data = (data - dataset.min()) / (dataset.max() - dataset.min())  # scale

                trained_neural_net = pickle.load(open('trained_neural_net', 'rb'))  # load trained neural net from file
                print()
                print('Female' if trained_neural_net.predict(data)[0] == 0 else 'Male')
        elif option == '3':
            print('\nExiting...')
            exit_ = True
        else:
            print('\nInvalid option. Please try again...')
