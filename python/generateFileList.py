import pandas as pd
import pickle

dataFolderPrefix = '../'
Folder = 'data20170130/'
Filename = 'driving_log.csv'
drivingLogFile = dataFolderPrefix + Folder + Filename
offset = 0.2
data = pd.read_csv(drivingLogFile, header=0,
                   names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])

speed_all = data.speed.values
X_all = []
y_all = []
for i in range(len(speed_all)):
    if float(speed_all[i]) < 20: continue
    # Load center image
    #imageFileName = dataFolderPrefix + Folder + data.center.values[i]
    imageFileName = data.center.values[i]
    angle = float(data.angle.values[i])
    X_all.append(imageFileName)
    y_all.append(angle)
    print(imageFileName)
    # Load left image
    #imageFileName = dataFolderPrefix + Folder + data.left.values[i].strip()
    imageFileName = data.left.values[i].strip()
    angle = float(data.angle.values[i] + offset)
    X_all.append(imageFileName)
    y_all.append(angle)
    print(imageFileName)
    # Load right image
    #imageFileName = dataFolderPrefix + Folder + data.right.values[i].strip()
    imageFileName = data.right.values[i].strip()
    angle = float(data.angle.values[i] - offset)
    X_all.append(imageFileName)
    y_all.append(angle)
    print(imageFileName)

data = {"fileNames": X_all, "label": y_all}
with open('../preprocessedData/data20170130OFFSET=' +str(offset) + '.p', 'wb') as f:
    pickle.dump(data, f)
