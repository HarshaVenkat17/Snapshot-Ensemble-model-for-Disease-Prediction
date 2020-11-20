from sklearn.metrics import accuracy_score
from keras.models import load_model
import numpy
import pandas as pd
from numpy import array
from numpy import argmax
 
# load models 
def load_models(models):
    all_models = []
    for i in range(models):
        filename = 'snapshot_model_' + str(i + 1) + '.h5'
        model = load_model(filename)
        print(">Loaded model %d"%(i+1))
        # add model to a list of models
        all_models.append(model)
    return all_models

def pred(members, X_test):
    predictions = [model.predict(X_test) for model in members]
    predictions = array(predictions)
    # sum across ensemble members
    summed = numpy.sum(predictions, axis=0)
    # argmax across classes to get predictions
    y_pred = argmax(summed, axis=1)
    return y_pred

# evaluate members in ensemble
def evaluate_models(members, X_test, y_test):
    y_pred=pred(members, X_test)
    return accuracy_score(y_test, y_pred)

# load dataset
data = pd.read_csv('Training.csv')

#preprocessing
data=data.drop_duplicates()
data.reset_index(drop=True,inplace=True)
X = data.iloc[:,:132]
y = data.iloc[:, 132]
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 0)

# load models 
members = load_models(5)
print('Loaded %d models' % len(members))

# evaluate model 
ensemble_score = evaluate_models(members, X_test, y_test)
print("Ensemble Accuracy: %.2f"%(ensemble_score*100))

single_row=pd.DataFrame([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).transpose()
single_pred=pred(members, single_row)
print("Predicted Disease:",*labelencoder_y.inverse_transform(single_pred))
