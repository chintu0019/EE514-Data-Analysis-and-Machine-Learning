import os
import glob
from glob import iglob 

rootdir = '../dataset/preprocessed'

""" for subdir, dirs, files in os.walk(rootdir):
    print(dirs) """

for f in iglob('../dataset/preprocessed/*', recursive=False):
    if os.path.isdir(f):
        print(f)
