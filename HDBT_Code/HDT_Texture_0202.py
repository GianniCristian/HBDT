# -*- coding: cp1252 -*-
import numpy as np
from copy import deepcopy
from skimage.feature import greycomatrix, greycoprops
import cv2
import osgeo.gdal, gdal
from osgeo.gdalconst import *
import skimage

from utils import Bar

def read_image(input_img):

    inb = osgeo.gdal.Open(input_img, GA_ReadOnly)
    data = inb.ReadAsArray().astype('float32')

    return data

def linear_quantization_1band_tmp(input_image,quantization_factor,low_tail=1.49,high_tail=97.8):
    '''Quantization of all the input bands cutting the tails of the distribution
    
    :param input_band_list: list of 2darrays (list of 2darrays)
    :param quantization_factor: number of levels as output (integer)
    :returns:  list of values corresponding to the quantized bands (list of 2darray)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 12/05/2014
    '''
    q_factor = quantization_factor - 1
    inmatrix = input_image.reshape(-1)
    out = np.bincount(inmatrix)
    tot = inmatrix.shape[0]
    freq = (out.astype(np.float32)/float(tot))*100 #frequency for each value
    cumfreqs = np.cumsum(freq)
    first = np.where(cumfreqs>low_tail)[0][0] #define occurrence limits for the distribution
    last = np.where(cumfreqs>high_tail)[0][0]
    temp_image = deepcopy(input_image)
    temp_image[np.where(temp_image>last)] = last
    temp_image[np.where(temp_image<first)] = first
    k1 = float(q_factor)/float((last-first)) #k1 term of the quantization formula
    k2 = np.ones(temp_image.shape)-k1*first*np.ones(temp_image.shape) #k2 term of the quantization formula
    out_matrix = np.floor(temp_image*k1+k2) #take the integer part
    out_matrix2 = out_matrix-np.ones(out_matrix.shape)
    out_matrix2.astype(np.uint8)
    temp_image = None
    
    return out_matrix2

def NDSV_plusBand(data):

    if data.ndim == 2:
        print 'NDSV is not possible'
        return
    n_bands = data.shape[0]
    result = np.zeros(((n_bands*(n_bands-1))/2,data.shape[1],data.shape[2]))
    h = 0
    for i in np.arange(0,n_bands+1):
        j = i
        while j != n_bands:
            if i != j:
                result[h] = (data[j][:][:].astype('float32') - data[i][:][:].astype('float32'))/\
                            (data[j][:][:].astype('float32') + data[i][:][:].astype('float32'))
                h += 1
            j +=1
    result_plus4band = np.zeros((result.shape[0]+data.shape[0],data.shape[1],data.shape[2]))
    result_plus4band[0:data.shape[0]] = data[0:data.shape[0],:,:]
    result_plus4band[data.shape[0]:] = result
    
    return result_plus4band

def Data_list_labels(data,mask,labels):

    data_list = []
    data_list_class = []

    if data.ndim != 2:
        for id_class in labels:
            indici = np.where(mask == id_class)
            for i in np.arange(0,len(indici[0])):
                data_list.append(data[0:data.shape[0],indici[0][i],indici[1][i]])
                data_list_class.append(id_class)
    else:
        for id_class in labels:
            indici = np.where(mask == id_class)
            for i in np.arange(0,len(indici[0])):
                data_list.append(data[indici[0][i],indici[1][i]])
                data_list_class.append(id_class)

    vect_list = np.vstack(data_list)
    vect_list_class =  np.vstack(data_list_class)

    return vect_list,vect_list_class

def Morph_Rec_plusBand(data,window_1,window_2,window_3,window_4):

    if data.ndim != 2:
        bright = np.mean(data,axis=0)        
    else:
        bright = deepcopy(data)
    
    se = [cv2.getStructuringElement(cv2.MORPH_RECT,(window_1,window_1)),\
          cv2.getStructuringElement(cv2.MORPH_RECT,(window_2,window_2)),\
          cv2.getStructuringElement(cv2.MORPH_RECT,(window_3,window_3)),\
          cv2.getStructuringElement(cv2.MORPH_RECT,(window_4,window_4))]
    if data.ndim != 2:
        morph_data = np.zeros((8,data.shape[1],data.shape[2]))
        morph_data_rec = np.zeros((8,data.shape[1],data.shape[2]))
    else:
        morph_data = np.zeros((8,data.shape[0],data.shape[1]))
        morph_data_rec = np.zeros((8,data.shape[0],data.shape[1]))

    for i in np.arange(0,4):
        morph_data[i] = cv2.dilate(bright,se[i])

    for i in np.arange(4,8):
        morph_data[i] = cv2.erode(bright,se[i-4])

    for i in np.arange(0,4):
        morph_data_rec[i] = skimage.morphology.reconstruction(morph_data[i],bright,method='erosion')

    for i in np.arange(4,8):
        morph_data_rec[i] = skimage.morphology.reconstruction(morph_data[i],bright,method='dilation')

    if data.ndim != 2:
        result = np.zeros((6+data.shape[0],data.shape[1],data.shape[2]))
        result[0:data.shape[0]] = data[0:data.shape[0],:,:]
        result[data.shape[0]:data.shape[0]+3] = morph_data_rec[1:4][:][:] - morph_data_rec[0:3][:][:]
        result[data.shape[0]+3:data.shape[0]+6] = morph_data_rec[5:8][:][:] - morph_data_rec[4:7][:][:]
    else:
        result = np.zeros((7,data.shape[0],data.shape[1]))
        result[0:1] = data[:,:]
        result[1:4] = morph_data_rec[1:4][:][:] - morph_data_rec[0:3][:][:]
        result[4:7] = morph_data_rec[5:8][:][:] - morph_data_rec[4:7][:][:]
        
    return result

def GLCM_min_all_plusBand(data,window_1,window_2,window_3,window_4,q_factor=64,prop='contrast'):

    if data.ndim == 2:
        result = np.zeros((20,data.shape[0],data.shape[1]))
        bright = deepcopy(data)
    else:
        result = np.zeros((20,data.shape[1],data.shape[2]))
        bright = np.mean(data,axis=0)
    bright = bright.astype('int64') 
    bright_q = linear_quantization_1band_tmp(bright,q_factor)
    length_bar = bright_q.shape[0]
    bright_q = np.pad(bright_q,window_4/2,mode='reflect')
    status_glcm = Bar(length_bar, status="\t GLCM")
    for i in xrange(window_4/2,bright_q.shape[0]-window_4/2):
        status_glcm(i+1)
        for j in xrange(window_4/2,bright_q.shape[1]-window_4/2):
            patch = bright_q[i-(window_1/2):i+(window_1/2)+1, j - (window_1/2):j+(window_1/2)+1]            
            glcm_greycoma = greycomatrix(patch, [1], [0,np.pi/2,np.pi/4,np.pi*(3/4)], q_factor, symmetric=False, normed=True)
            glcm_1_cont = np.min(greycoprops(glcm_greycoma, prop='contrast'))
            glcm_1_diss = np.min(greycoprops(glcm_greycoma, prop='dissimilarity'))
            glcm_1_homo = np.min(greycoprops(glcm_greycoma, prop='homogeneity'))
            glcm_1_ASM = np.min(greycoprops(glcm_greycoma, prop='ASM'))
            glcm_1_corr = np.min(greycoprops(glcm_greycoma, prop='correlation'))

            
            patch = bright_q[i-(window_2/2):i+(window_2/2)+1, j - (window_2/2):j+(window_2/2)+1]            
            glcm_greycoma = greycomatrix(patch, [1], [0,np.pi/2,np.pi/4,np.pi*(3/4)], q_factor, symmetric=False, normed=True)
            glcm_2_cont = np.min(greycoprops(glcm_greycoma, prop='contrast'))
            glcm_2_diss = np.min(greycoprops(glcm_greycoma, prop='dissimilarity'))
            glcm_2_homo = np.min(greycoprops(glcm_greycoma, prop='homogeneity'))
            glcm_2_ASM = np.min(greycoprops(glcm_greycoma, prop='ASM'))
            glcm_2_corr = np.min(greycoprops(glcm_greycoma, prop='correlation'))
            
            patch = bright_q[i-(window_3/2):i+(window_3/2)+1, j - (window_3/2):j+(window_3/2)+1]
            glcm_greycoma = greycomatrix(patch, [1], [0,np.pi/2,np.pi/4,np.pi*(3/4)], q_factor, symmetric=False, normed=True)
            glcm_3_cont = np.min(greycoprops(glcm_greycoma, prop='contrast'))
            glcm_3_diss = np.min(greycoprops(glcm_greycoma, prop='dissimilarity'))
            glcm_3_homo = np.min(greycoprops(glcm_greycoma, prop='homogeneity'))
            glcm_3_ASM = np.min(greycoprops(glcm_greycoma, prop='ASM'))
            glcm_3_corr = np.min(greycoprops(glcm_greycoma, prop='correlation'))

            patch = bright_q[i-(window_4/2):i+(window_4/2)+1, j - (window_4/2):j+(window_4/2)+1]
            glcm_greycoma = greycomatrix(patch, [1], [0,np.pi/2,np.pi/4,np.pi*(3/4)], q_factor, symmetric=False, normed=True)
            glcm_4_cont = np.min(greycoprops(glcm_greycoma, prop='contrast'))
            glcm_4_diss = np.min(greycoprops(glcm_greycoma, prop='dissimilarity'))
            glcm_4_homo = np.min(greycoprops(glcm_greycoma, prop='homogeneity'))
            glcm_4_ASM = np.min(greycoprops(glcm_greycoma, prop='ASM'))
            glcm_4_corr = np.min(greycoprops(glcm_greycoma, prop='correlation'))

            result[0][i-window_4/2][j-window_4/2]= glcm_1_cont
            result[1][i-window_4/2][j-window_4/2]= glcm_1_diss
            result[2][i-window_4/2][j-window_4/2]= glcm_1_homo
            result[3][i-window_4/2][j-window_4/2]= glcm_1_ASM
            result[4][i-window_4/2][j-window_4/2]= glcm_1_corr
            result[5][i-window_4/2][j-window_4/2]= glcm_2_cont
            result[6][i-window_4/2][j-window_4/2]= glcm_2_diss
            result[7][i-window_4/2][j-window_4/2]= glcm_2_homo
            result[8][i-window_4/2][j-window_4/2]= glcm_2_ASM
            result[9][i-window_4/2][j-window_4/2]= glcm_2_corr
            result[10][i-window_4/2][j-window_4/2]= glcm_3_cont
            result[11][i-window_4/2][j-window_4/2]= glcm_3_diss
            result[12][i-window_4/2][j-window_4/2]= glcm_3_homo
            result[13][i-window_4/2][j-window_4/2]= glcm_3_ASM
            result[14][i-window_4/2][j-window_4/2]= glcm_3_corr
            result[15][i-window_4/2][j-window_4/2]= glcm_4_cont
            result[16][i-window_4/2][j-window_4/2]= glcm_4_diss
            result[17][i-window_4/2][j-window_4/2]= glcm_4_homo
            result[18][i-window_4/2][j-window_4/2]= glcm_4_ASM
            result[19][i-window_4/2][j-window_4/2]= glcm_4_corr

    if data.ndim != 2:
        result_fin = np.zeros((20+data.shape[0],data.shape[1],data.shape[2]))
        result_fin[0:data.shape[0]] = data[:,:,:]
        result_fin[data.shape[0]:] = result[:,:,:]
    else:
        result_fin = np.zeros((21,data.shape[0],data.shape[1]))
        result_fin[0:1] = data[:,:]
        result_fin[1:] = result[:,:,:]
    
    return result_fin
