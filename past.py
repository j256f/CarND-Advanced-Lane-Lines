import numpy as np
import cv2

class past():

    def __init__(self, chunk_size=10):

        #self.w_h = image_sz

        self.last_frames = [] #np.zeros((1,2,20))  
        self.how_many_frames = chunk_size
        for i in range(1, self.how_many_frames):
            self.last_frames.append(0)
       
    def past_frames(self, preprocessImage1, chunk_size=10):

        self.last_frames.append((preprocessImage1))
        #print(len(self.last_frames))
        #exit()
        return self.last_frames[-chunk_size:]

                
        


    






