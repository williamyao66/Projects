import pandas as pd 
import os 

from tensorflow.keras.models import load_model


tr = pd.read_csv('./Digit-Recognizer/train.csv')
te = pd.read_csv('./Digit-Recognizer/test.csv')
x_te = te.values.copy().astype('float64')

final_model = load_model(os.getcwd() + '/model/')
predictions = final_model.predict(x_te)
test_predictions = np.argmax(predictions, axis=1)