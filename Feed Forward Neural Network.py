#######################################################
###############                         ###############
###############  @Nimesha Muthunayake   ###############
###############  Reg: 209359G           ###############
###############  Date:07-29-2020        ###############
###############  M.Sc 2020              ###############
###############                         ###############
#######################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from tensorflow.keras.utils import plot_model
%matplotlib inline

# Loading the dataset
dataset = pd.read_csv('/content/drive/My Drive/Nimesha/crx.csv')
dataset.shape
dataset.dtypes


# Preparing the dataset
for col in ['A1', 'A2','A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'A14', 'A16']:
    dataset[col] = pd.Categorical(dataset[col])
    dataset[col] = dataset[col].cat.codes

dataset.dtypes   
dataset.head() 


# apply scaler 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset = pd.DataFrame(scaler.fit_transform(dataset))

#dataset.describe()


X=dataset.iloc[:,0:15].values   #0:15
Y=dataset.iloc[:,15:16].values

np.random.seed(42) # Makes the random numbers

#Create the model generator function
#Deciding the numbr of layers and the number of neurons 

depth = 3
neuronCount = [16, 9, 1]                                
activationFunction = ['sigmoid', 'relu', 'sigmoid']           
lossFunction = 'binary_crossentropy'                                
regularization = tf.keras.regularizers.l2(0)                      


def create_model(verbose=False):

  model = tf.keras.models.Sequential()  
  if verbose:
        print('Configuration: ',neuronCount)  
  model.add(tf.keras.layers.Dense(neuronCount[0], input_dim=15, activation = activationFunction[0], kernel_regularizer=regularizerFunc))     
  for x in range(1, depth-1):
      model.add(tf.keras.layers.Dense(neuronCount[x], activation = activationFunction[x],kernel_regularizer=regularizerFunc)) 
  model.add(tf.keras.layers.Dense(neuronCount[depth-1], activation = activationFunction[depth-1]))  # Output layer    
  model.compile(loss = lossFunction , optimizer = 'adam' , metrics = ['accuracy'] ) 
        
  return model

depth = 1
neuronCount = [10, 8, 1]                                 
activationFunction = ['sigmoid', 'relu', 'sigmoid']            
lossFunction = 'binary_crossentropy'                               
regularizerFunc = tf.keras.regularizers.l2(0)                      

#Create the model
model=create_model()


def evaluate_model(verbose=False):
    K=5 #Number of folds
    f1_scores = []

    for train_index,test_index in StratifiedKFold(K).split(X, Y):      #Starting the 5 fold cross validation 
        x_train,x_test=X[train_index],X[test_index]
        y_train,y_test=Y[train_index],Y[test_index]

        model=create_model(verbose)
        model.fit(x_train, y_train,epochs=100, verbose=0)
        evl = model.evaluate(x_test,y_test, verbose=0)
        
        if verbose:
            print('Evaluated the model',evl)  

        y_pred = np.where(model.predict(x_test) > 0.5, 1, 0)
        f1 = f1_score(y_test, y_pred , average="macro")

        if verbose:
            print('F1 score is ', f1)
        
        f1_scores.append(f1)    
    return np.mean(f1_scores)


###########  Start testing with chaning the first layer neuron count  ##########
###########  Depth of network = 2
###########  Number of neuron = 15

depth = 2
neuronCount = [25, 1]                                 
activationFunction = ['sigmoid', 'sigmoid']           
lossFunction = 'binary_crossentropy'                                
regularizerFunc = tf.keras.regularizers.l2(0)                      

avg_f1 = evaluate_model(True)
print('Depth 2---> Average F1', avg_f1)


###########  Start testing with chaning the second layer neuron count  ##########
###########  Depth of network = 3
###########  Number of neuron = 18 , 9 

depth = 3
neuronCount = [18,9, 1]                                
activationFunction = ['relu', 'relu', 'relu']           
lossFunction = 'mean_squared_error'                                
regularizerFunc = tf.keras.regularizers.l2(0)   

avg_f1 = evaluate_model(True)
print('Depth 3---> Average F1', avg_f1)

########### Different lambda values for L1 regularization**
########### Depth of network = 2
########### Number of neuron = 15,1

depth = 2
neuronCount = [15,1]                                 
activationFunction = ['tanh', 'sigmoid']           
lossFunction = 'binary_crossentropy'                               
regularizerFunc = tf.keras.regularizers.l1(0)                      

for i in range(-5,5):
    regularizerFunc = tf.keras.regularizers.l1(10**i)
    print("'L1 Regularization : L1 with lambda : % 10.5f , Mean F1 score : % 10.5f" %(10**i, evaluate_model()))



########### Different lambda values for L2 regularization**
########### Depth of network = 2
########### Number of neuron = 15,1

depth = 2
neuronCount = [15,1]                                
activationFunction = ['tanh', 'sigmoid']           
lossFunction = 'binary_crossentropy'                                
regularizerFunc = tf.keras.regularizers.l2(0)                      

for i in range(-5,5):
    regularizerFunc = tf.keras.regularizers.l2(10**i)
    print("'L2 Regularization : L2 with lambda : % 10.5f , Mean F1 score : % 10.5f" %(10**i, evaluate_model()))