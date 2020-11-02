import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from preprocess_data_val import preprocess
import argparse
from tensorflow.keras.models import load_model


filename = 'IPC.h5'
max_length = 1431
max_num_words = 23140
nclasses = 451

x_val, y_val = preprocess(max_length, max_num_words)

model = load_model(filename)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

y_val = to_categorical(np.asarray(y_val),
                       num_classes=nclasses).astype(np.float16)

score = model.evaluate(x_val, y_val, verbose=0)

print('Evaluating Model ...')

print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

print("Prediction Started ...")
y_pred = model.predict(x_val)

# Generate arg maxes for predictions
classes = np.argmax(y_pred, axis = 1)
print('Predicted :', classes)
act = np.argmax(y_val, axis =1)
print('Actuals :', act)
