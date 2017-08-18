import numpy as np

from models.squeeze import squeeze as maia
from keras.models import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

FILE_I_END = 46

WIDTH = 250
HEIGHT = 173
LR = 1e-3
EPOCHS = 30

SAVED = 'saved/'

MODEL_NAME = 'squeeze'
PREV_MODEL = 'squeeze'

LOAD_MODEL = True

wl = 0
sl = 0
al = 0
dl = 0

wal = 0
wdl = 0
sal = 0
sdl = 0
nkl = 0

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

if LOAD_MODEL:
    model = load_model(SAVED + PREV_MODEL + '.h5')
    print('We have loaded a previous model!!!!')
else:
    model = maia()
# iterates through the training files

try:
    file_name = 'training_data.npy'
    # full file info
    train = np.load(file_name)

    X = np.array([i[0] for i in train])
    Y = [i[1] for i in train]

    model.fit(X, Y,
              batch_size=64,
              epochs=EPOCHS,
              validation_split=0.1,
              callbacks=[
                  ModelCheckpoint(SAVED + MODEL_NAME + '.h5', save_best_only=True)]
              )

except Exception as e:
    print(e)
