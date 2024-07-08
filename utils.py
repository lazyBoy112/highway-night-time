import os

def get_negative_description():
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negative'):
            f.write('negative/'+filename+'\n')

def fix_des(w, h):
    file = open('pos.txt', 'r')
    while(True):
        info = file.readline()
        if not info :
            break
        info = info.strip().split(' ')
        for i, v in enumerate(info):
            if not i:
                continue
            info[i] = int(v)
        if info[2] + info[4] > w or info[3] + info[5]> h:
            print(info)


# get_negative_description()
fix_des(1280, 960)
