# -*- coding: cp1252 -*-
import osgeo.gdal, gdal
from osgeo.gdalconst import *
import time
import sys
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from itertools import combinations
import collections

def Conf_Matr_2arr(predict,truth):
    
    labels = np.unique(truth)

    #Remove Background Class
    if truth.min() == 0:
        labels = np.delete(labels,0)

    if truth.ndim == 2:
        predict = predict.ravel()
        truth = truth.ravel()
    #Compute confusion matrix
    try:
        cm = confusion_matrix(predict,truth,labels)
    except:                      
        raise ValueError('Wrong Input Image Size.')
    n_class = np.size(cm[0])

    cm_per = np.zeros((n_class,n_class))
    for i in np.arange(n_class):
        cm_per[:,i]= (cm[:,i]/float(np.sum(cm[:,i])))*100

    ###Compute statistical parameters

    #Cohen's kappa
    po =  0
    rowsum = 0
    for i in np.arange(n_class):
        po += float(cm[i,i])
        rowsum += float(np.sum(cm[:,i]))

    petot = 0
    for i in np.arange(n_class):
        petot += (float(np.sum(cm[:,i]))* float(np.sum(cm[i,:])))/rowsum
         
    kappa = (po-petot)/(rowsum-petot)

    #Prod and User's Accuracy
    prod_acc = np.zeros((1,n_class))
    user_acc = np.zeros((1,n_class))
    for i in np.arange(n_class):
        prod_acc[0][i]= cm_per[i,i]
        if np.sum(cm[i,:])== 0:
            user_acc[0][i] = 0
        else:
            user_acc[0][i]= (float(cm[i,i])/ np.sum(cm[i,:]))*100

    #Average Accuracy
    aver_acc = np.sum(prod_acc)/n_class

    #Overall Accuracy
    total = 0
    for i in np.arange(n_class):
        total += cm[i,i]
        
    overall = (total/rowsum)*100

    score = overall
    return overall,aver_acc,kappa,user_acc,prod_acc,cm

def Find_Class_kappa_labels_UserAndProd(dictionary,labels,kappa_min = 0.5):

    prod_valore_max = 0
    user_valore_max = 0
    temp_kappa = 0 
    classe_val_max = 0
    for i in dictionary:
        if (np.amax(dictionary[i][4]+dictionary[i][3]) >= prod_valore_max) and (dictionary[i][2] >=kappa_min):
            if np.amax(dictionary[i][4]+dictionary[i][3]) == prod_valore_max:
                if np.amax(dictionary[i][2]) > temp_kappa:
                    temp_kappa = dictionary[i][2]
                    var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                    prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                    classifier = i
                    if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                        classe_val_max = labels[var[1][0]]
                    else:
                        user_valore_max = 0
                        for h in var[1]:
                            if dictionary[i][3][0][h] > user_valore_max:
                                user_valore_max = dictionary[i][3][0][h]
                                classe_val_max = labels[h]
            else:
                temp_kappa = dictionary[i][2]
                var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                classifier = i
                if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                    classe_val_max = labels[var[1][0]]
                else:
                    user_valore_max = 0
                    for h in var[1]:
                        if dictionary[i][3][0][h] > user_valore_max:
                            user_valore_max = dictionary[i][3][0][h]
                            classe_val_max = labels[h]
        
    return classifier, classe_val_max

def Find_Class_kappa_labels_UserAndProd_param(dictionary,labels,kappa_min = 0.5):

    prod_valore_max = 0
    user_valore_max = 0
    temp_kappa = 0 
    classe_val_max = 0
    for i in dictionary:
        if (np.amax(dictionary[i][4]+dictionary[i][3]) >= prod_valore_max) and (dictionary[i][2] >=kappa_min):
            if np.amax(dictionary[i][4]+dictionary[i][3]) == prod_valore_max:
                if np.amax(dictionary[i][2]) > temp_kappa:
                    temp_kappa = dictionary[i][2]
                    var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                    prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                    classifier = i
                    lista_feature = dictionary[i][6]
                    if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                        classe_val_max = labels[var[1][0]]
                    else:
                        user_valore_max = 0
                        for h in var[1]:
                            if dictionary[i][3][0][h] > user_valore_max:
                                user_valore_max = dictionary[i][3][0][h]
                                classe_val_max = labels[h]
            else:
                temp_kappa = dictionary[i][2]
                var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                classifier = i
                lista_feature = dictionary[i][6]
                if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                    classe_val_max = labels[var[1][0]]
                else:
                    user_valore_max = 0
                    for h in var[1]:
                        if dictionary[i][3][0][h] > user_valore_max:
                            user_valore_max = dictionary[i][3][0][h]
                            classe_val_max = labels[h]
        
    return classifier, classe_val_max, lista_feature


def Find_Param_labels_UserAndProd(classifier,dictionary,label_class,n_bands,train_list,train_list_class,GT_list,GT_list_class,kappa_min = 0.5):
    
    list_feat = np.arange(n_bands,train_list.shape[1])
    count = len(list_feat)
    
    nested_dict = lambda: collections.defaultdict(nested_dict)
    d = nested_dict()

    base = np.arange(n_bands)
    #Caso originale
    result_temp = dictionary[classifier](train_list,train_list_class,GT_list,GT_list_class)
    d[tuple(list_feat)] = result_temp

    while count > 1:
        combo = combinations(list_feat,count-1)
        for i in combo:
            temp_lungh = n_bands
            temp_feat = np.concatenate((base,i))
            result_temp = dictionary[classifier](train_list[:,temp_feat],train_list_class,GT_list[:,temp_feat],GT_list_class)
            d[i] = result_temp
        classifier_sub, classe_val_max = Find_Param_kappa_labels_UserAndProd(d,label_class,kappa_min = 0.5)
        if classifier_sub == 0:
            classifier_sub = list_feat
        if set(classifier_sub)!= set(list_feat):
            list_feat = classifier_sub
        else:
            break
        count -= 1
    
    return list_feat, classe_val_max

def Find_Param_kappa_labels_UserAndProd(dictionary,labels,kappa_min = 0.5):

    prod_valore_max = 0
    user_valore_max = 0
    temp_kappa = 0 
    classe_val_max = 0
    for i in dictionary:
        if (np.amax(dictionary[i][4]+dictionary[i][3]) >= prod_valore_max) and (dictionary[i][2] >=kappa_min):
            if np.amax(dictionary[i][4]+dictionary[i][3]) == prod_valore_max:
                if np.amax(dictionary[i][2]) > temp_kappa:
                    temp_kappa = dictionary[i][2]
                    var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                    prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                    classifier = i
                    if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                        classe_val_max = labels[var[1][0]]
                    else:
                        user_valore_max = 0
                        for h in var[1]:
                            if dictionary[i][3][0][h] > user_valore_max:
                                user_valore_max = dictionary[i][3][0][h]
                                classe_val_max = labels[h]
            else:
                temp_kappa = dictionary[i][2]
                var = np.where(dictionary[i][4]+dictionary[i][3] == np.amax(dictionary[i][4]+dictionary[i][3]))
                prod_valore_max = dictionary[i][4][var[0][0]][var[1][0]]+dictionary[i][3][var[0][0]][var[1][0]]
                classifier = i
                if np.count_nonzero(dictionary[i][4][var[0][:],var[1][:]]) == 1:
                    classe_val_max = labels[var[1][0]]
                else:
                    user_valore_max = 0
                    for h in var[1]:
                        if dictionary[i][3][0][h] > user_valore_max:
                            user_valore_max = dictionary[i][3][0][h]
                            classe_val_max = labels[h]
    try:
        return classifier, classe_val_max
    except:
        print '0','0'
        return 0,0
        
