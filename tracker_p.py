import numpy as np
import cv2

class tracker_p():

    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):

        self.recent_centers = []

        self.window_width = Mywindow_width

        self.window_height = Mywindow_height

        self.margin = Mymargin

        self.ym_per_pix = My_ym

        self.xm_per_pix = My_xm

        self.smooth_factor = Mysmooth_factor


    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin


        window_centroids = []
        window = np.ones(window_width)
        
        # the follwing code finds the first set of center of lanes and will act as anchor 
        # It wil add all white pixels in the 1/5 bottom and from 1/4 to 1/2 for the left lanes
        # and for the right lane it will add from 1/2 to 3/4
        
        l_sum = np.sum(warped[int(4*warped.shape[0]/5):,int(1*warped.shape[1]/4):int(warped.shape[1]/2)], axis=0)
        
        r_sum = np.sum(warped[int(4*warped.shape[0]/5):,int(warped.shape[1]/2):3*int(warped.shape[1]/4)], axis=0)
        
        # r_center, l_center are x posotions where the maximum amount of white pixels are clustered
        # If one lane center is very faint compared to the other, it will take values of the other lane        # this is done by subtracting or adding 500 pixels to the x value were  max 


        if max(l_sum)*20 < max(r_sum):

            r_center = np.argmax(r_sum)-window_width/2+int(warped.shape[1]/2)
            l_center = r_center - 500 
            window_centroids.append((l_center,r_center))


        elif max(r_sum)*20 < max(l_sum):

            l_center = np.argmax(l_sum)-window_width/2+int(1*warped.shape[1]/4)
            r_center = l_center + 500
            window_centroids.append((l_center,r_center))
 
        else:

            l_center = np.argmax(l_sum)-window_width/2+int(1*warped.shape[1]/4)
            r_center = np.argmax(r_sum)-window_width/2+int(warped.shape[1]/2)
            window_centroids.append((l_center,r_center))
                
        
        for level in range(1,(int(warped.shape[0]/window_height))):

            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:],axis=0)
            
            # this loop will build the lane from the anchor values
            # it will find the x position where the maximum values of white is clustered
            # the position is to be found within limits acording to the "margin" value 
       
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin, warped.shape[1]))
            l_add = image_layer[l_min_index:l_max_index]
            
                                
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
            r_add = image_layer[r_min_index:r_max_index]
            
            # as for the anchor values, the stronger lane helps the weak lane

            if max(l_add)*20 < max(r_add):

                r_center = np.argmax(image_layer[r_min_index:r_max_index])+r_min_index-offset
                l_center = r_center - 500  
                window_centroids.append((l_center,r_center))
            
            elif max(r_add)*20 < max(l_add):

                l_center = np.argmax(image_layer[l_min_index:l_max_index])+l_min_index-offset 
                r_center = l_center + 500 
                window_centroids.append((l_center,r_center))
            
            else:    
                l_center = np.argmax(image_layer[l_min_index:l_max_index])+l_min_index-offset
                r_center = np.argmax(image_layer[r_min_index:r_max_index])+r_min_index-offset
                window_centroids.append((l_center,r_center))



            #print(window_centroids)
            #print(self.recent_centers)
        

        
        self.recent_centers.append(((window_centroids)))
        #print(window_centroids[0][0])
        #print(window_centroids[0][1])
        #print(len(window_centroids))
        #print(self.recent_centers)
        #print(np.average(self.recent_centers[-self.smooth_factor:], axis=0))
        #exit()                
        #print(np.subtract(l_center,r_center))        
        
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

                
        


    






