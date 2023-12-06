from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
import tensorflow_addons as tfa
from keras.regularizers import L1
from keras.callbacks import EarlyStopping
import numpy as np

# fit the model
def model_fit(X, y, X_val, y_val, penalty, learning_rate, decay_rate, momentum, batch_size):

  model = Sequential()
  model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=L1(penalty), bias_regularizer=L1(penalty)))
  model.add(BatchNormalization())
  model.add(Dense(1, kernel_regularizer=L1(penalty), bias_regularizer=L1(penalty)))

# add early stop
  earlystop = EarlyStopping(monitor='val_r_square', mode='max', patience=10, verbose=1, restore_best_weights=True)
  
# Compile model

  sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
  model.compile(loss='mse', optimizer=sgd, metrics= [tfa.metrics.RSquare()])
  model.fit(X, y, epochs=100, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val), callbacks=earlystop)
  return model

# make an ensemble prediction
def ensemble_predict(members, X_test_scaled):
  
# make predictions
  yhats = [model.predict(X_test_scaled) for model in members]
  yhats = np.array(yhats)
# take average across ensemble members
  result = np.mean(yhats, axis=0)
# get the predicted result
  return result

from sklearn.metrics import r2_score
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, X_test_scaled, y_test_scaled):
# select a subset of members
  subset = members[:n_members]
# make prediction
  yhat = ensemble_predict(subset, X_test_scaled)
# calculate accuracy
  return r2_score(y_test_scaled, yhat)