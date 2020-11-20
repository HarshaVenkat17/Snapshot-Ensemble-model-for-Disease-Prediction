#Snapshot model for Disease prediction
import pandas as pd
import numpy as np
from keras import backend
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
 

class SnapshotEnsemble(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, decay, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.alpha  = lrate_max
        self.loss   = decay
        self.lrates = list()
 
    # calculate learning rate for each epoch
    def annealing(self, epoch, alpha, loss):
        alpha=alpha* 1/(1 + loss * epoch)
        return alpha
 
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs={}):
        # calculate learning rate
        lr = self.annealing(epoch, self.alpha, self.loss)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # save the value 
        self.lrates.append(lr)
 
    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle =int(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))
    
    
# read the dataset
data = pd.read_csv('Training.csv')

#preprocessing 
data=data.drop_duplicates()
data.reset_index(drop=True,inplace=True)
X = data.iloc[:,:132]
y = data.iloc[:, 132]
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y = to_categorical(y)

# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state = 5)

# create an ANN with 1 input layer, 1 hidden and 1 output layer
model = Sequential()
model.add(Dense(70, input_dim=132, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(41, activation='softmax'))

#create an optimizer instance and compile the model
opt=SGD(momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#create 10 cycles each with 5 epochs
n_epochs = 50
n_cycles = n_epochs//10

#create a callback for snapshot model
ca = SnapshotEnsemble(n_epochs, n_cycles, 0.1,0.1)

# fit model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, verbose=0, callbacks=[ca])
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print('Train: %.2f, Test: %.2f' % (train_acc, test_acc))

# plot to show learning accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()

y_hat=np.argmax(model.predict(X_test),axis=1)
y_test=np.argmax(y_test,axis=1)
print(accuracy_score(y_test, y_hat))