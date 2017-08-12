import numpy as np

from utils.models import inception_v3 as googlenet

FILE_I_END = 46

WIDTH = 180
HEIGHT = 133
LR = 1e-3
EPOCHS = 30

SAVED = 'saved/'

MODEL_NAME = 'test'
PREV_MODEL = 'test'

LOAD_MODEL = False

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

model = googlenet(HEIGHT, WIDTH, 3, LR, output=9, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(SAVED + PREV_MODEL)
    print('We have loaded a previous model!!!!')

# iterates through the training files

for e in range(EPOCHS):
    print("--------------------- EPOCH: {} --------------------".format(e))
    try:
        file_name = 'training_data.npy'
        # full file info
        train_data = np.load(file_name)

        train = train_data[:-50]
        test = train_data[-50:]

        X = np.array([i[0] for i in train]).reshape(-1, HEIGHT, WIDTH, 3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, HEIGHT, WIDTH, 3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)

        print('SAVING MODEL!')
        model.save(SAVED + MODEL_NAME)

    except Exception as e:
        print(e)
