import os
import h5py
import numpy as np
from PIL import Image

# Lacinak irni, hogy: 
# kene dron datase
# mi van meg
# beszeltem lucaval
# aug. 9-en lesz a kovetkezo meeting, addig az image segmentationnel fogok foglalkozni

def get(df, what):
    return np.array(df.get(what))

def get_one_line(ds, idx):
    image_path = get(ds, 'camera_data/color_left')[idx]
    segmentation_path = get(ds,'camera_data/segmentation')[idx]
    gyro = get(dset,'imu/gyroscope')[idx]
    accel = get(dset,'imu/accelerometer')[idx]
    position = get(dset, 'groundtruth/position')[idx]
    velocity = get(dset, 'groundtruth/velocity')[idx]
    return (image_path, segmentation_path, gyro, accel, position, velocity)

# open file 
f1 = h5py.File('./sensor_records.hdf5')
# list of trajectories
print(f1.keys())
# access one 
dset = f1['trajectory_4006']
print(dset.keys())
# imu/gyroscope data
print(dset)
imu = get(dset,'imu/gyroscope')

camera = get(dset,'camera_data/color_left')
segmentation = get(dset,'camera_data/segmentation')

print(camera.shape)
print(segmentation.shape)

#img = Image.open(camera[0])
#img.show()

data = get_one_line(dset, 3)
print(data)
