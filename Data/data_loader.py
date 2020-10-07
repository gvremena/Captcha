import json
import math
import os
import sys
import numpy as np

data_path = "data.npy"

def linedist(a, b, t):
    return abs((-a*t[0] + t[1] - b))/(math.sqrt(a*a + 1))

def flatten(l):
    return [item for sublist in l for item in sublist]

def total_distance(l):
    d = 0
    for i in range(1, len(l)):
        d += np.sqrt((l[i][0] - l[i - 1][0])*(l[i][0] - l[i - 1][0]) + (l[i][1] - l[i - 1][1])*(l[i][1] - l[i - 1][1]))
    return d

def distance(t1, t2):
    return np.sqrt((t1[0] - t2[0])*(t1[0] - t2[0]) + (t1[1] - t2[1])*(t1[1] - t2[1]))

def print_data():
    data = np.load(data_path)
    print(data)

def load_data(input_json):
    new_data = input_json
    data = np.load(data_path)

    correct = new_data["correct"]
    total = new_data["total"]
    dist = new_data["dist"]
    clicks = new_data["clicks"]
    time = new_data["timer"]
    mouse = new_data["mouse"]
    mouse_flat = [i for sublist in mouse for i in sublist]

    line_distances =[[0 for j in range(len(mouse[i]))] for i in range(len(mouse))]
    pauses = 0

    for i in range(len(mouse)):
        start = mouse[i][0]
        end = mouse[i][len(mouse[i]) - 1]
        a = (end[1] - start[1])/(end[0] - start[0])
        b = start[1] - a*start[0]
        for j in range(len(mouse[i])):
            line_distances[i][j] = linedist(a, b, mouse[i][j])

    mouse_pauses = [[]]
    for i in range(len(mouse_flat)):
        diff = mouse_flat[i][2] - mouse_flat[i - 1][2]
        if diff >= 300:
            pauses += 1
            mouse_pauses += [[mouse_flat[i]]]
        else:
            mouse_pauses[len(mouse_pauses) - 1] = mouse_pauses[len(mouse_pauses) - 1] + [mouse_flat[i]]

    velocity = [[]]*len(mouse_pauses)
    acceleration = [[]]*len(mouse_pauses)
    for k in range(len(mouse_pauses)):
        f = mouse_pauses[k]
        for i in range(1, len(f)):
            if(len(f)) > 1:
                dtime = (f[i][2] - f[i - 1][2])/1000
                ddist = distance(f[i], f[i - 1])
                velocity[k] = velocity[k] + [ddist/dtime]
                if i > 1:
                    acceleration[k] = acceleration[k] + [np.abs(ddist/dtime - velocity[k][i - 2])/dtime]
            elif len(f) <= 2:
                acceleration[k] = [0]
            elif len(f) <= 1:
                velocity[k] = [0]

    l = np.array([])
    l = np.append(l, correct/total) #
    l = np.append(l, pauses) # number of pauses
    l = np.append(l, np.var(dist)) # distance between center
    l = np.append(l, time[0]/1000) # start time
    l = np.append(l, time[len(time) - 1]/1000) # end time
    l = np.append(l, np.var(np.diff(time))/1000) # time between click
    l = np.append(l, np.var(flatten(velocity)))
    l = np.append(l, np.var(flatten(line_distances)))
    l = np.append(l, np.abs(np.var(flatten(acceleration))))
    #l = np.append(l, normal/len(mouse)) # % normal
    #l = np.append(l, uniform/len(mouse)) # % uniform

    if len(data) == 0 or (not any(np.equal(data, l).all(1))):
        if len(data) == 0:
            data = np.array([l])
        else:
            data = np.append(data, np.array([l]), axis=0)

        np.save(data_path, data)

    #if (delete == True):
    #    os.remove(fpath)

    return l

if __name__ == "__main__":
    fpath = os.path.abspath("D:/Download/result.txt")

    if not os.path.exists(data_path):
        np.save("data.npy", np.empty((0, 6)))

    data = np.load(data_path)
    print(data)

