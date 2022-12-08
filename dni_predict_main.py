import pandas as pd
import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop, Adagrad, Adamax, Nadam


_Curr_Dir = os.getcwd()
Data_Dir = os.path.join(_Curr_Dir, 'Data')
print(Data_Dir)


def combine_arrays(df, input_array, dni_array):
    ghi  = df['GHI'].values
    hours = df['Hour'].values
    minutes = df['Minute'].values
    Days = df['Day'].values
    solar_zenith = df['Solar Zenith Angle'].values
    pressure = df['Pressure'].values
    dni = df['DNI'].values
    Temperature = df['Temperature'].values
    ar = np.array([ghi,Temperature,Days, hours, minutes, solar_zenith, pressure]).T
    input_array = np.concatenate((input_array, ar))
    dni_array = np.concatenate((dni_array, np.array(dni).T))
    return input_array, dni_array


def main_model(acti, opti, epochs):
    input_size = 7    #change if var is removed
    input_array = np.array([], dtype=np.float64).reshape(0,input_size)
    dni_array = np.array([], dtype=np.float64)
    
    
    for path in glob(Data_Dir+'/Train'):
        for file in glob(path+'/psm_'+'*'):
            print(file)
            df = pd.read_csv(file)
            #count_nan = df.isnull().sum()
            #print("Nan value count is:", count_nan)
            input_array, dni_array = combine_arrays(df, input_array, dni_array)
    keras.backend.clear_session()
    tf.keras.backend.clear_session()
    keras.backend.clear_session()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph

    acti = acti

    model_dni = keras.Sequential()
    model_dni.add(keras.layers.Dense(240, activation=acti))
    model_dni.add(layers.Dense(120, activation=acti))
    model_dni.add(layers.Dense(90, activation=acti))
    model_dni.add(layers.Dense(60, activation=acti))
    model_dni.add(layers.Dense(1, activation=acti))
   
    opti = opti
    #Compile and train
    model_dni.compile(optimizer=opti, 
                  loss='mse',
                  metrics=['mae',keras.metrics.RootMeanSquaredError()])
    history_dni = model_dni.fit(input_array, dni_array, epochs=epochs, batch_size=64)
    dot_img_file = 'model_1.png'
    tf.keras.utils.plot_model(model_dni, to_file=dot_img_file, show_shapes=True)
    score_dni_train = model_dni.evaluate(input_array, dni_array, verbose=1)
    return model_dni, history_dni

def plotting(model_history):
    plt.plot(model_history.history['mae'])
    plt.plot(model_history.history['root_mean_squared_error'])
    val = 0
    if val == 1:
        plt.plot(model_history.history['val_mae'])
        plt.plot(model_history.history['val_root_mean_squared_error'])
        plt.legend(['MAE', 'RMSE', 'val_MAE', 'val_RMSE'])
    else:
        plt.legend(['MAE', 'RMSE'])
        
    plt.title('MAE and RMSE per epoch')
    plt.ylabel('Error (W/m^2)')
    plt.xlabel('Epoch')
    plt.savefig("loss_graph.png")
    plt.show()
    
def calc_change(pred, actual):
    sum_pred = sum(pred)
    sum_act = sum(actual)
    perc_change = (((abs(sum_pred - sum_act)/sum_act))*100)
    print("Testing - Percentage Change:", perc_change)


def testing(model_dni, plot = False, plot_data = False):
    test_df = pd.read_csv(os.path.join(Data_Dir+'/Test', 'psm_CO_Boulder2021.csv'))

    input_size = 7
    test_input_array = np.array([], dtype=np.float64).reshape(0,input_size)
    test_dni_array = np.array([], dtype=np.float64)

    test_input_array, test_dni_array = combine_arrays(test_df, test_input_array, test_dni_array)

    score_dni = model_dni.evaluate(test_input_array, test_dni_array, verbose=1)
    
    if plot_data == True:
        plt.plot(test_dni_array[0:720])
        plt.title('DNI of January 2021')
        plt.ylabel('Direct Normal Irradiance (W/m^2)')
        plt.xlabel('Hour Number')
        plt.savefig("data.png")
        plt.show()
        
        plt.plot(test_dni_array[0:24], '-o', color='red')
        plt.title('DNI of 1st January 2021')
        plt.ylabel('Direct Normal Irradiance (W/m^2)')
        plt.xlabel('Hour Number')
        plt.savefig("data_day.png")
        plt.show()
    
    preds = model_dni.predict(test_input_array)
    if plot == True:
        plt.plot(preds[0:720])
        plt.plot(test_dni_array[0:720])
        plt.legend(['Predicted DNI', 'Actual DNI'])
        plt.title('Actual vs Predicted')
        plt.ylabel('Direct Normal Irradiance (W/m^2)')
        plt.xlabel('Hour Number')
        plt.savefig("Prediction.png")
        plt.show()
    

    from sklearn.metrics import mean_absolute_error as mae
    
    mae_dni = mae(test_dni_array,preds)
    print("Test MAE", mae_dni)
    calc_change(preds, test_dni_array)
    
    return mae_dni




if __name__ == "__main__":
    
    lr = 0.001
    model, history = main_model(acti = "LeakyReLU", opti = Adam(learning_rate=lr), epochs = 120)
    plotting(history)
    testing(model, plot = True, plot_data= True)
  
    
  
    ############################################################
    #                                                          #
    #            Hyperparameter Optimization (Manual Loop)     #
    #                                                          #
    ############################################################
    
    #Loop has to be adjusted for each parameter/combination
    
    
    # #acti = ["relu", "LeakyReLU"]
    # #lr = [0.001, 0.0001, 0.00001]
    # lr = 0.001
    # #opti_labels = ["Adam", "RMSprop", "Adagrad","Adadelta", "Adamax","Nadam"]
    # #opti = [Adam(learning_rate=lr), RMSprop(learning_rate=lr),Adagrad(learning_rate=lr), Adadelta(learning_rate=lr)
    # #        ,Adamax(learning_rate=lr),Nadam(learning_rate=lr)]

    
    # #opti = Adam(learning_rate=lr)
    # #param = opti
    # acti = "LeakyReLU"
    # n_nodes = [5,10,20,40] #         ,240,300,360,480]
    
    # test_MAEs = []
    # for i in range(len(n_nodes)):
    #     model, history = main_model(acti, Adam(learning_rate=lr), n_nodes[i])
    #     plt.plot(history.history['mae'], label = str(n_nodes[i]))
    #     test_MAEs.append(testing(model))
    # plt.legend(loc = 'upper right')  
    # plt.ylabel('Mean Absolute Error (W/m^2)')
    # plt.xlabel('Epoch')
    # plt.title("Training: Dense_Nodes_5-2")
    # plt.savefig("Train_Dense_Nodes_5-2.png")
    # plt.show()
    
    

    # xs = range(len(test_MAEs))
    # plt.bar(xs,test_MAEs)
    # # HERE tell pyplot which labels correspond to which x values
    # plt.ylabel('Test Mean Absolute Error (W/m^2)')
    # plt.title("Testing: Dense_Nodes_5-2")
    # plt.xticks(xs,n_nodes)
    # plt.savefig("Test_Dense_Nodes_5-2.png")
    # plt.show()
















