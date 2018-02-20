import h5py
import os
import numpy as np
import io

num = 3
in_file = 'flowers.hdf5'
out_file = 'flowers_val.hdf5'
in_group = 'test'
out_group = 'val'

fi = h5py.File(in_file)[in_group]
fo = h5py.File(out_file)
fo = fo.create_group(out_group)

cnt = 0
for name in fi:
  data = fi[name]
  new = fo.create_group(name)
  new.create_dataset('name', data=data['name'])
  new.create_dataset('img', data=data['img'])
  new.create_dataset('embeddings', data=data['embeddings'])
  new.create_dataset('class', data=data['class'])
  new.create_dataset('txt', data=data['txt'])
  cnt += 1
  if(cnt >= num):
    break
    
