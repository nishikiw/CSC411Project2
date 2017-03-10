from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import hashlib
import re
from shutil import copy

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


def download_image(actor_list):
    create_dir("cropped")
    create_dir("uncropped")
    
    testfile = urllib.URLopener()
    faces_files = ["facescrub_actors.txt", "facescrub_actresses.txt"]
    for face_file in faces_files:
        for a in actor_list:
            name = a.split()[1].lower()
            i = 0
            for line in open(face_file):
                if a in line:
                    filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                    timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                    if not os.path.isfile("uncropped/"+filename):
                        continue
                    else:
                        try:
                            # check the hash of the files
                            file = open("uncropped/" + filename, "rb")
                            print(filename)
                            hashcode = line.split()[6]
                            imagecode = hashlib.sha256(file.read()).hexdigest()
                            if (imagecode != hashcode):
                                print("Hash Error")
                            else:
                                im = Image.open("uncropped/"+filename)
                                box = (line.split()[5])
                                x1 = int(box.split(",")[0])
                                y1 = int(box.split(",")[1])
                                x2 = int(box.split(",")[2])
                                y2 = int(box.split(",")[3])
                                crop_border = (x1,y1,x2,y2)
                                im2 = im.crop(crop_border)
                                im3 = imresize(im2, (32, 32))
                                im4 = rgb2gray(im3)
                                imsave("cropped/" + filename, im4)
                            file.close()
                        except:
                            pass
                    i += 1
    print("Finished downloading!")


def split_set(act, train_size, val_size, test_size):
    create_dir("part7_training")
    create_dir("part7_validation")
    create_dir("part7_test")
    dict = build_act_dict()
    for name in act:
        lastname = name.split()[1].lower()
        np.random.seed(0)
        set = np.random.choice(dict[lastname], train_size+val_size+test_size, replace=False)
        for i in range(0, train_size):
            copy("cropped/"+set[i], "part7_training")
        for i in range(train_size, train_size+val_size):
            copy("cropped/"+set[i], "part7_validation")
        for i in range(train_size+val_size, train_size+val_size+test_size):
            copy("cropped/"+set[i], "part7_test")
    print("Finish splitting sets")
            

def build_act_dict():
    dict = {}
    for file in os.listdir("cropped"):
        name = re.sub(r"\d", "", file.split(".")[0])
        if name in dict:
            dict[name].append(file)
        else:
            dict[name] = [file]
    return dict