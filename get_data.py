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
    for a in act:
        used_image = []
        name = a.split()[1].lower()
        split_one_set(name, train_size, "part7_training", used_image)
        split_one_set(name, val_size, "part7_validation", used_image)
        split_one_set(name, test_size, "part7_test", used_image)
    print("Finished splitting!")
        

def split_one_set(name, size, dest_file, used_image_no):
    counter = 0
    while (counter < size):
        r = random.randint(1, 200)
        if r in used_image_no:
            continue
        filename = name+str(r)+'.'+'jpg'
        try:
            im = imread("cropped/"+filename)
            imsave(dest_file+"/"+filename, im)
            used_image_no.append(r)
            counter += 1
        except:
            pass