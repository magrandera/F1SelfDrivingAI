import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

FILE_I_END = 61
LoadPrev = False

data_order = [i for i in range(2, FILE_I_END + 1)]
shuffle(data_order)

train_data = np.load('training/training_data-1.npy')

for count, i in enumerate(data_order):
    file_name = 'training/training_data-{}.npy'.format(i)
    # full file info
    load_data = np.load(file_name)
    train_data = np.concatenate((train_data, load_data))
    print('training_data-{}.npy'.format(i), len(train_data))

if LoadPrev:
    old = np.load('training_data.npy')
    train_data = np.concatenate((train_data, old))

df = pd.DataFrame(train_data)
Counter(df[1].apply(str))

w = []
s = []
a = []
d = []
wa = []
wd = []
sa = []
sd = []
nk = []


shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0, 0, 0, 0, 0, 0, 0]:
        w.append([img,choice])
    elif choice == [0, 1, 0, 0, 0, 0, 0, 0, 0]:
        s.append([img,choice])
    elif choice == [0, 0, 1, 0, 0, 0, 0, 0, 0]:
        a.append([img,choice])
    elif choice == [0, 0, 0, 1, 0, 0, 0, 0, 0]:
        d.append([img, choice])
    elif choice == [0, 0, 0, 0, 1, 0, 0, 0, 0]:
        wa.append([img, choice])
    elif choice == [0, 0, 0, 0, 0, 1, 0, 0, 0]:
        wd.append([img, choice])
    elif choice == [0, 0, 0, 0, 0, 0, 1, 0, 0]:
        sa.append([img, choice])
    elif choice == [0, 0, 0, 0, 0, 0, 0, 1, 0]:
        sd.append([img, choice])
    elif choice == [0, 0, 0, 0, 0, 0, 0, 0, 1]:
        nk.append([img, choice])
    else:
        print('no matches')

print("W: {}".format(len(w)))
print("S: {}".format(len(s)))
print("A: {}".format(len(a)))
print("D: {}".format(len(d)))
print("WA: {}".format(len(wa)))
print("WD: {}".format(len(wd)))
print("SA: {}".format(len(sa)))
print("SD: {}".format(len(sd)))
print("NK: {}".format(len(nk)))

balance = min(len(w), len(a), len(d), len(wa), len(wd), len(nk))

w = w[:2*balance]
s = s[:balance]
a = a[:balance]
d = d[:balance]
wa = wa[:round(1.2*balance)]
wd = wd[:round(1.2*balance)]
sa = sa[:balance]
sd = sd[:balance]
nk = nk[:balance]


final_data = w + s + a + d + wa + wd + sa + sd + nk
shuffle(final_data)

np.save('training_data.npy', final_data)