# -*- coding: cp1252 -*-
import osgeo.gdal, gdal
from osgeo.gdalconst import *
import sys
import numpy as np
import sklearn
from AccuracyAssesment_PRIN_2509 import *
from Classifiers_2509 import *
from sklearn.cross_validation import StratifiedShuffleSplit
from HDT_Texture_0202 import *
import copy
from utils import Bar

import argparse
import warnings

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    input_img = str(arg.input_img)
    truth_img = str(arg.truth_img)
    hdt_class_path = str(arg.hdt_class_path)
    output_txt = str(arg.output_txt)
    morph_win = arg.morph_win
    glcm_win = arg.glcm_win
    test_size_def = arg.test_size_def
    splitFeatures = arg.splitFeatures
    print input_img,truth_img,hdt_class_path,output_txt,morph_win,glcm_win,test_size_def,splitFeatures
    
    Classify_HBDT(input_img,
                  truth_img,
                  hdt_class_path,
                  output_txt,
                  morph_win,
                  glcm_win,
                  test_size_def,
                  splitFeatures)

def args():
    parser = argparse.ArgumentParser(description='HBDT')
    parser.add_argument("input_img", help="????")
    parser.add_argument("truth_img", help="????")
    parser.add_argument("hdt_class_path", help="????")
    parser.add_argument("output_txt", help="????")
    parser.add_argument("--morph_win", nargs=4, help="????")
    parser.add_argument("--glcm_win", nargs=4, help="????")
    parser.add_argument("test_size_def", help="????")
    parser.add_argument("splitFeatures", help="????")
    args = parser.parse_args()
    return args

def Classify_HBDT(input_img,truth_img,hdt_class_path,output_txt,morph_win,glcm_win,test_size_def,splitFeatures):
    
    test_size_def = float(test_size_def)
    splitFeatures = float(splitFeatures)
    
    morph_se_1 = int(morph_win[0])
    morph_se_2 = int(morph_win[1])
    morph_se_3 = int(morph_win[2])
    morph_se_4 = int(morph_win[3])

    glcm_win_1 = int(glcm_win[0])
    glcm_win_2 = int(glcm_win[1])
    glcm_win_3 = int(glcm_win[2])
    glcm_win_4 = int(glcm_win[3])

    #Fixed Parameters

    min_mlh = 0
    max_mlh = 1

    n_test = 1

    scaling = 1 #1 yes - 0 no

    #Read Input image
    inb = osgeo.gdal.Open(input_img, GA_ReadOnly)
    data = inb.ReadAsArray().astype('float32')
    geoTransform = inb.GetGeoTransform()
    proj = inb.GetProjection()
    n_bands = inb.RasterCount
    inb = None

    #Morph_Image
    morph = Morph_Rec_plusBand(data,morph_se_1,morph_se_2,morph_se_3,morph_se_4)
    ndsv = NDSV_plusBand(data)
    glcm = GLCM_min_all_plusBand(data,glcm_win_1,glcm_win_2,glcm_win_3,glcm_win_4)

    #Read Ground Truth image
    inb = osgeo.gdal.Open(truth_img, GA_ReadOnly)
    inband = inb.GetRasterBand(1)
    data_TrainTest = inband.ReadAsArray()
    inb = None

    #Se esiste lo zero lo elimino -> Background del GT
    labels = np.unique(data_TrainTest)
    if data_TrainTest.min() == 0:
        labels = np.delete(labels,0)

    #Intestazione file di testo

    fw = open(output_txt, "w")
    fw.write('\nOver Acc \tAver Acc \tKappa \tCM')
    fw.close()

    #SPLIT GT in Training and Test set

    if data.ndim != 2:
        data_train = np.zeros((data.shape[1],data.shape[2]))
        data_gt = np.zeros((data.shape[1],data.shape[2]))
    else:
        data_train = np.zeros((data.shape[0],data.shape[1]))
        data_gt = np.zeros((data.shape[0],data.shape[1]))

    #Ravel
    data_TrainTest_ravel = data_TrainTest.ravel()
    data_train_ravel = data_train.ravel()
    data_gt_ravel = data_gt.ravel()

    split = StratifiedShuffleSplit(data_TrainTest_ravel,n_test,test_size=test_size_def,random_state=0)

    for train_index, test_index in split:
        
        data_train_ravel[train_index] = data_TrainTest_ravel[train_index]
        data_gt_ravel[test_index] = data_TrainTest_ravel[test_index]
        n_test -= 1
        
        #Reshape
        if data.ndim != 2:
            data_train = data_train_ravel.reshape((data.shape[1],data.shape[2]))
            data_gt = data_gt_ravel.reshape((data.shape[1],data.shape[2]))
        else:
            data_train = data_train_ravel.reshape((data.shape[0],data.shape[1]))
            data_gt = data_gt_ravel.reshape((data.shape[0],data.shape[1]))
        
        #Training Sets

        train_list_nosc, train_list_class = Data_list_labels(data,data_train,labels)
        train_list_class = train_list_class[:,0]
        morph_train_nosc, morph_train_cls = Data_list_labels(morph,data_train,labels)
        morph_train_cls = morph_train_cls[:,0]
        glcm_train_nosc, glcm_train_cls = Data_list_labels(glcm,data_train,labels)
        glcm_train_cls = glcm_train_cls[:,0]
        
        #Test Sets

        GT_list_nosc, GT_list_class = Data_list_labels(data,data_gt,labels)
        morph_gt_nosc, morph_gt_cls= Data_list_labels(morph,data_gt,labels)
        glcm_gt_nosc, glcm_gt_cls= Data_list_labels(glcm,data_gt,labels)
        if ndsv != None:
            ndsv_train_nosc, ndsv_train_cls = Data_list_labels(ndsv,data_train,labels)
            ndsv_train_cls = ndsv_train_cls[:,0]
            ndsv_gt_nosc, ndsv_gt_cls= Data_list_labels(ndsv,data_gt,labels)
        
        morph_classifier = ['Morph_KNN','Morph_SVM','Morph_RF','Morph_MLH']    
        glcm_classifier = ['Glcm_KNN','Glcm_SVM','Glcm_RF','Glcm_MLH']
        ndsv_classifier = ['Ndsv_KNN','Ndsv_SVM','Ndsv_RF','Ndsv_MLH']
        
        #HDTress Construction

        class_leaf = []
        classifier_leaf = []
        classifier_param = []

        #Scaling
        
        std_scale = preprocessing.StandardScaler().fit(train_list_nosc)
        train_list = std_scale.transform(train_list_nosc)
        GT_list = std_scale.transform(GT_list_nosc)
        std_scale = preprocessing.StandardScaler().fit(morph_train_nosc)
        morph_train = std_scale.transform(morph_train_nosc)
        morph_gt = std_scale.transform(morph_gt_nosc)
        std_scale = preprocessing.StandardScaler().fit(glcm_train_nosc)
        glcm_train = std_scale.transform(glcm_train_nosc)
        glcm_gt = std_scale.transform(glcm_gt_nosc)
        #MLH case
        std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(train_list_nosc)
        train_list_mlh = std_scale.transform(train_list_nosc)
        GT_list_mlh = std_scale.transform(GT_list_nosc)
        std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(morph_train_nosc)
        morph_train_mlh = std_scale.transform(morph_train_nosc)
        morph_gt_mlh = std_scale.transform(morph_gt_nosc)
        std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(glcm_train_nosc)
        glcm_train_mlh = std_scale.transform(glcm_train_nosc)
        glcm_gt_mlh = std_scale.transform(glcm_gt_nosc)
        if ndsv != None:
            std_scale = preprocessing.StandardScaler().fit(ndsv_train_nosc)
            ndsv_train = std_scale.transform(ndsv_train_nosc)
            ndsv_gt = std_scale.transform(ndsv_gt_nosc)
            std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(ndsv_train_nosc)
            ndsv_train_mlh = std_scale.transform(ndsv_train_nosc)
            ndsv_gt_mlh = std_scale.transform(ndsv_gt_nosc)
        

        fw = open(output_txt, "a")

        #Per la scelta del classificatore, ovvero della classe vincitrice.

        label_class = np.unique(data_TrainTest)
        if data_TrainTest.min() == 0:
            label_class = np.delete(label_class,0)
        label_class = list(label_class)

        if ndsv != None:
            dzn_subset = {'KNN': KNNeighbors, 'SVM': SVMachine, 'RF': RForest, 'MLH':NB_MLHood,\
                          'Morph_KNN': KNNeighbors,'Morph_SVM': SVMachine,'Morph_RF': RForest,\
                          'Morph_MLH': NB_MLHood,\
                          'Glcm_KNN': KNNeighbors,'Glcm_SVM': SVMachine,'Glcm_RF': RForest,\
                          'Glcm_MLH': NB_MLHood,\
                          'Ndsv_KNN': KNNeighbors,'Ndsv_SVM': SVMachine,'Ndsv_RF': RForest,\
                          'Ndsv_MLH': NB_MLHood}
        else:
            dzn_subset = {'KNN': KNNeighbors, 'SVM': SVMachine, 'RF': RForest, 'MLH':NB_MLHood,\
                          'Morph_KNN': KNNeighbors,'Morph_SVM': SVMachine,'Morph_RF': RForest,\
                          'Morph_MLH': NB_MLHood,\
                          'Glcm_KNN': KNNeighbors,'Glcm_SVM': SVMachine,'Glcm_RF': RForest,\
                          'Glcm_MLH': NB_MLHood}

        length_bar = len(labels)
        status_tree = Bar(length_bar, status="\t HBDT construction")
        for i in labels[1:]:
            status_tree(i+1)
            if splitFeatures == 2:

                list_feat_tmp = []
                result_KNN = KNNeighbors_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_feat_tmp)
                result_SVM = SVMachine_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_feat_tmp)
                result_RF = RForest_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_feat_tmp)
                result_MLH = NB_MLHood_param(train_list_mlh,train_list_class,GT_list_mlh,GT_list_class,n_bands,list_feat_tmp)
                
                classifier_tmp = 'Morph_KNN'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_KNN = KNNeighbors_param(morph_train,morph_train_cls,morph_gt,morph_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Morph_SVM'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_SVM = SVMachine_param(morph_train,morph_train_cls,morph_gt,morph_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Morph_RF'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_RF = RForest_param(morph_train,morph_train_cls,morph_gt,morph_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Morph_MLH'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,morph_train_mlh,morph_train_cls,morph_gt_mlh,morph_gt_cls)
                result_Morph_MLH = NB_MLHood_param(morph_train_mlh,morph_train_cls,morph_gt_mlh,morph_gt_cls,n_bands,list_feat_tmp)

                classifier_tmp = 'Glcm_KNN'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_KNN = KNNeighbors_param(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Glcm_SVM'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_SVM = SVMachine_param(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Glcm_RF'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_RF = RForest_param(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls,n_bands,list_feat_tmp)
                classifier_tmp = 'Glcm_MLH'
                list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,glcm_train_mlh,glcm_train_cls,glcm_gt_mlh,glcm_gt_cls)
                result_Glcm_MLH = NB_MLHood_param(glcm_train_mlh,glcm_train_cls,glcm_gt_mlh,glcm_gt_cls,n_bands,list_feat_tmp)

                if ndsv != None:
                    classifier_tmp = 'Ndsv_KNN'
                    list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_KNN = KNNeighbors_param(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls,n_bands,list_feat_tmp)
                    classifier_tmp = 'Ndsv_SVM'
                    list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_SVM = SVMachine_param(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls,n_bands,list_feat_tmp)
                    classifier_tmp = 'Ndsv_RF'
                    list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_RF = RForest_param(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls,n_bands,list_feat_tmp)
                    classifier_tmp = 'Ndsv_MLH'
                    list_feat_tmp, classe_val_max_tmp =  Find_Param_labels_UserAndProd(classifier_tmp,dzn_subset,label_class,n_bands,ndsv_train_mlh,ndsv_train_cls,ndsv_gt_mlh,ndsv_gt_cls)
                    result_Ndsv_MLH = NB_MLHood_param(ndsv_train_mlh,ndsv_train_cls,ndsv_gt_mlh,ndsv_gt_cls,n_bands,list_feat_tmp)

            else:
                result_KNN = KNNeighbors(train_list,train_list_class,GT_list,GT_list_class)
                result_SVM = SVMachine(train_list,train_list_class,GT_list,GT_list_class)
                result_RF = RForest(train_list,train_list_class,GT_list,GT_list_class)
                result_MLH = NB_MLHood(train_list_mlh,train_list_class,GT_list_mlh,GT_list_class)
                
                result_Morph_KNN = KNNeighbors(morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_SVM = SVMachine(morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_RF = RForest(morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                result_Morph_MLH = NB_MLHood(morph_train_mlh,morph_train_cls,morph_gt_mlh,morph_gt_cls)
                
                result_Glcm_KNN = KNNeighbors(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_SVM = SVMachine(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_RF = RForest(glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                result_Glcm_MLH = NB_MLHood(glcm_train_mlh,glcm_train_cls,glcm_gt_mlh,glcm_gt_cls)

                if ndsv != None:
                    result_Ndsv_KNN = KNNeighbors(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_SVM = SVMachine(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_RF = RForest(ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    result_Ndsv_MLH = NB_MLHood(ndsv_train_mlh,ndsv_train_cls,ndsv_gt_mlh,ndsv_gt_cls)

            if ndsv != None:
                dizionario = {'KNN': result_KNN, 'SVM': result_SVM, 'RF': result_RF, 'MLH':result_MLH,\
                              'Morph_KNN': result_Morph_KNN,'Morph_SVM': result_Morph_SVM,'Morph_RF': result_Morph_RF,\
                              'Morph_MLH': result_Morph_MLH,\
                              'Glcm_KNN': result_Glcm_KNN,'Glcm_SVM': result_Glcm_SVM,'Glcm_RF': result_Glcm_RF,\
                              'Glcm_MLH': result_Glcm_MLH,\
                              'Ndsv_KNN': result_Ndsv_KNN,'Ndsv_SVM': result_Ndsv_SVM,'Ndsv_RF': result_Ndsv_RF,\
                              'Ndsv_MLH': result_Ndsv_MLH}
            else:
                dizionario = {'KNN': result_KNN, 'SVM': result_SVM, 'RF': result_RF, 'MLH':result_MLH,\
                              'Morph_KNN': result_Morph_KNN,'Morph_SVM': result_Morph_SVM,'Morph_RF': result_Morph_RF,\
                              'Morph_MLH': result_Morph_MLH,\
                              'Glcm_KNN': result_Glcm_KNN,'Glcm_SVM': result_Glcm_SVM,'Glcm_RF': result_Glcm_RF,\
                              'Glcm_MLH': result_Glcm_MLH}
                

            if splitFeatures == 2:
                classifier, classe_val_max, list_feat = Find_Class_kappa_labels_UserAndProd_param(dizionario,label_class,kappa_min = 0.5)
            else:
                classifier, classe_val_max = Find_Class_kappa_labels_UserAndProd(dizionario,label_class,kappa_min = 0.5)
            
            if classifier in morph_classifier:
                if splitFeatures == 1:
                    if classifier == 'Morph_MLH':
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,morph_train_mlh,morph_train_cls,morph_gt_mlh,morph_gt_cls)
                    else:
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,morph_train,morph_train_cls,morph_gt,morph_gt_cls)
                    indici = np.where(morph_train_cls == classe_val_max)
                    indici_test = np.where(morph_gt_cls == classe_val_max)
                else:
                    indici = np.where(morph_train_cls == classe_val_max)
                    indici_test = np.where(morph_gt_cls == classe_val_max)
            elif classifier in glcm_classifier:
                if splitFeatures == 1:
                    if classifier == 'Glcm_MLH':
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,glcm_train_mlh,glcm_train_cls,glcm_gt_mlh,glcm_gt_cls)
                    else:
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,glcm_train,glcm_train_cls,glcm_gt,glcm_gt_cls)
                    indici = np.where(glcm_train_cls == classe_val_max)
                    indici_test = np.where(glcm_gt_cls == classe_val_max)
                else:
                    indici = np.where(glcm_train_cls == classe_val_max)
                    indici_test = np.where(glcm_gt_cls == classe_val_max)
            elif classifier in ndsv_classifier:
                if splitFeatures == 1:
                    if classifier == 'Ndsv_MLH':
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,ndsv_train_mlh,ndsv_train_cls,ndsv_gt_mlh,ndsv_gt_cls)
                    else:
                        list_feat, classe_val_max =  Find_Param_labels_UserAndProd(classifier,dzn_subset,label_class,n_bands,ndsv_train,ndsv_train_cls,ndsv_gt,ndsv_gt_cls)
                    indici = np.where(ndsv_train_cls == classe_val_max)
                    indici_test = np.where(ndsv_gt_cls == classe_val_max)
                else:
                    indici = np.where(ndsv_train_cls == classe_val_max)
                    indici_test = np.where(ndsv_gt_cls == classe_val_max)
            else:
                indici = np.where(train_list_class == classe_val_max)
                indici_test = np.where(GT_list_class == classe_val_max)
            
            #Rimozione Training Set
            train_list = np.delete(train_list,indici,axis=0)
            train_list_mlh = np.delete(train_list_mlh,indici,axis=0)
            train_list_class = np.delete(train_list_class,indici,axis=0)
            morph_train = np.delete(morph_train,indici,axis=0)
            morph_train_mlh = np.delete(morph_train_mlh,indici,axis=0)
            morph_train_cls = np.delete(morph_train_cls,indici,axis=0)
            glcm_train = np.delete(glcm_train,indici,axis=0)
            glcm_train_mlh = np.delete(glcm_train_mlh,indici,axis=0)
            glcm_train_cls = np.delete(glcm_train_cls,indici,axis=0)
            

            #Rimozione Test Set
            GT_list = np.delete(GT_list,indici_test,axis=0)
            GT_list_mlh = np.delete(GT_list_mlh,indici_test,axis=0)
            GT_list_class = np.delete(GT_list_class,indici_test,axis=0)
            morph_gt = np.delete(morph_gt,indici_test,axis=0)
            morph_gt_mlh = np.delete(morph_gt_mlh,indici_test,axis=0)
            morph_gt_cls = np.delete(morph_gt_cls,indici_test,axis=0)
            glcm_gt = np.delete(glcm_gt,indici_test,axis=0)
            glcm_gt_mlh = np.delete(glcm_gt_mlh,indici_test,axis=0)
            glcm_gt_cls = np.delete(glcm_gt_cls,indici_test,axis=0)
            
            if ndsv != None:
                ndsv_train = np.delete(ndsv_train,indici,axis=0)
                ndsv_train_mlh = np.delete(ndsv_train_mlh,indici,axis=0)
                ndsv_train_cls = np.delete(ndsv_train_cls,indici,axis=0)
                ndsv_gt = np.delete(ndsv_gt,indici_test,axis=0)
                ndsv_gt_mlh = np.delete(ndsv_gt_mlh,indici_test,axis=0)
                ndsv_gt_cls = np.delete(ndsv_gt_cls,indici_test,axis=0)
            

            
            label_class.remove(classe_val_max)
            class_leaf.append(classe_val_max)
            classifier_leaf.append(classifier)
            if splitFeatures == 1 or splitFeatures == 2:
                classifier_param.append(tuple(list_feat))
            fw.write('\nSTEP:\t'+str(i-1)+'\tClassifier:\t'+str(classifier)+'\tClass:\t'+str(classe_val_max))
        
        fw.close()
        
        #END HDTress Construction
        
        #Classification of the data
        if data.ndim != 2:
            class_data = np.zeros((data.shape[1],data.shape[2]),dtype = int)
        else:
            class_data = np.zeros((data.shape[0],data.shape[1]),dtype = int)
            
        #TUTTI GLI INDICI
        
        indici = np.where(data_gt >= 0)
        
        split = 30000
        
        length_bar = len(indici[0])/split+1
        status_class = Bar(length_bar, status="\t Image Classification")
        for j in xrange(0, len(indici[0]),split):
            status_class(j/split+2)
            last_class = np.unique(data_TrainTest)
            if data_TrainTest.min() == 0:
                last_class = np.delete(last_class,0)
            last_class = list(last_class)

            Data_list_nosc = []
            Morph_Data_list_nosc = []
            Glcm_Data_list_nosc = []
            Ndsv_Data_list_nosc = []

            if (j+split > len(indici[0])):
                split = -(j-len(indici[0]))
            for i in np.arange(j,j+split):
                if (i % 80000 == 0):
                   print i
                #Data set
                
                if data.ndim != 2:
                    Data_list_nosc.append(data[0:int(data.shape[0]),indici[0][i],indici[1][i]])
                else:
                    Data_list_nosc.append(data[indici[0][i],indici[1][i]])
                Morph_Data_list_nosc.append(morph[0:int(morph.shape[0]),indici[0][i],indici[1][i]])
                Glcm_Data_list_nosc.append(glcm[0:int(glcm.shape[0]),indici[0][i],indici[1][i]])
                if ndsv != None:
                    Ndsv_Data_list_nosc.append(ndsv[0:int(ndsv.shape[0]),indici[0][i],indici[1][i]])

            #Data set
            Data_list_nosc = np.vstack(Data_list_nosc)
            Morph_Data_list_nosc = np.vstack(Morph_Data_list_nosc)
            Glcm_Data_list_nosc = np.vstack(Glcm_Data_list_nosc)
            if ndsv != None:
                Ndsv_Data_list_nosc = np.vstack(Ndsv_Data_list_nosc)
            
            #Training Sets
            
            train_list_nosc, train_list_class = Data_list_labels(data,data_train,labels)
            train_list_class = train_list_class[:,0]
            morph_train_nosc, morph_train_cls = Data_list_labels(morph,data_train,labels)
            morph_train_cls = morph_train_cls[:,0]
            glcm_train_nosc, glcm_train_cls = Data_list_labels(glcm,data_train,labels)
            glcm_train_cls = glcm_train_cls[:,0]
            
            if ndsv != None:
                ndsv_train_nosc, ndsv_train_cls = Data_list_labels(ndsv,data_train,labels)
                ndsv_train_cls = ndsv_train_cls[:,0]
            
            #Scaling

            std_scale = preprocessing.StandardScaler().fit(train_list_nosc)
            train_list = std_scale.transform(train_list_nosc)
            Data_list = std_scale.transform(Data_list_nosc)
            std_scale = preprocessing.StandardScaler().fit(morph_train_nosc)
            morph_train = std_scale.transform(morph_train_nosc)
            Morph_Data_list = std_scale.transform(Morph_Data_list_nosc)
            std_scale = preprocessing.StandardScaler().fit(glcm_train_nosc)
            glcm_train = std_scale.transform(glcm_train_nosc)
            Glcm_Data_list = std_scale.transform(Glcm_Data_list_nosc)
            
            #MLH case
            std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(train_list_nosc)
            train_list_mlh = std_scale.transform(train_list_nosc)
            Data_list_mlh = std_scale.transform(Data_list_nosc)
            std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(morph_train_nosc)
            morph_train_mlh = std_scale.transform(morph_train_nosc)
            Morph_Data_list_mlh = std_scale.transform(Morph_Data_list_nosc)
            std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(glcm_train_nosc)
            glcm_train_mlh = std_scale.transform(glcm_train_nosc)
            Glcm_Data_list_mlh = std_scale.transform(Glcm_Data_list_nosc)
            if ndsv != None:
                std_scale = preprocessing.StandardScaler().fit(ndsv_train_nosc)
                ndsv_train = std_scale.transform(ndsv_train_nosc)
                Ndsv_Data_list = std_scale.transform(Ndsv_Data_list_nosc)
                std_scale = preprocessing.MinMaxScaler(feature_range=(min_mlh, max_mlh)).fit(ndsv_train_nosc)
                ndsv_train_mlh = std_scale.transform(ndsv_train_nosc)
                Ndsv_Data_list_mlh = std_scale.transform(Ndsv_Data_list_nosc)

            if ndsv != None:
                dzn_class = {'KNN': KNNeighbors_class,'SVM': SVMachine_class,'RF':  RForest_class,'MLH':NB_MLHood_class,\
                             'Morph_KNN': KNNeighbors_class,'Morph_SVM': SVMachine_class,'Morph_RF': RForest_class,\
                             'Morph_MLH': NB_MLHood_class,\
                             'Glcm_KNN': KNNeighbors_class,'Glcm_SVM': SVMachine_class,'Glcm_RF': RForest_class,\
                             'Glcm_MLH': NB_MLHood_class,\
                             'Ndsv_KNN': KNNeighbors_class,'Ndsv_SVM': SVMachine_class,'Ndsv_RF': RForest_class,\
                             'Ndsv_MLH': NB_MLHood_class}
                
                dzn_class_param = {'KNN': KNNeighbors_class_param,'SVM': SVMachine_class_param,'RF':  RForest_class_param,'MLH':NB_MLHood_class_param,\
                             'Morph_KNN': KNNeighbors_class_param,'Morph_SVM': SVMachine_class_param,'Morph_RF': RForest_class_param,\
                             'Morph_MLH': NB_MLHood_class_param,\
                             'Glcm_KNN': KNNeighbors_class_param,'Glcm_SVM': SVMachine_class_param,'Glcm_RF': RForest_class_param,\
                             'Glcm_MLH': NB_MLHood_class_param,\
                             'Ndsv_KNN': KNNeighbors_class_param,'Ndsv_SVM': SVMachine_class_param,'Ndsv_RF': RForest_class_param,\
                             'Ndsv_MLH': NB_MLHood_class_param}
            else:
                dzn_class = {'KNN': KNNeighbors_class,'SVM': SVMachine_class,'RF':  RForest_class,'MLH':NB_MLHood_class,\
                             'Morph_KNN': KNNeighbors_class,'Morph_SVM': SVMachine_class,'Morph_RF': RForest_class,\
                             'Morph_MLH': NB_MLHood_class,\
                             'Glcm_KNN': KNNeighbors_class,'Glcm_SVM': SVMachine_class,'Glcm_RF': RForest_class,\
                             'Glcm_MLH': NB_MLHood_class}
                
                dzn_class_param = {'KNN': KNNeighbors_class_param,'SVM': SVMachine_class_param,'RF':  RForest_class_param,'MLH':NB_MLHood_class_param,\
                             'Morph_KNN': KNNeighbors_class_param,'Morph_SVM': SVMachine_class_param,'Morph_RF': RForest_class_param,\
                             'Morph_MLH': NB_MLHood_class_param,\
                             'Glcm_KNN': KNNeighbors_class_param,'Glcm_SVM': SVMachine_class_param,'Glcm_RF': RForest_class_param,\
                             'Glcm_MLH': NB_MLHood_class_param}

            if data.ndim != 2:
                class_data_tmp = np.zeros((data.shape[1],data.shape[2]),dtype=int)
            else:
                class_data_tmp = np.zeros((data.shape[0],data.shape[1]),dtype=int)

            for i in np.arange(0,len(labels)-1):

                labels_train = np.unique(train_list_class)
                cls = class_leaf[i]
                classifier = classifier_leaf[i]
                if splitFeatures == 1 or splitFeatures == 2 :
                    param = classifier_param[i]

                if splitFeatures == 1 or splitFeatures == 2 :
                    if classifier in morph_classifier:
                        if classifier == 'Morph_MLH':
                            vettore_class = dzn_class_param[classifier_leaf[i]](morph_train_mlh,morph_train_cls,Morph_Data_list_mlh,n_bands,param)
                        else:
                            vettore_class = dzn_class_param[classifier_leaf[i]](morph_train,morph_train_cls,Morph_Data_list,n_bands,param)
                    elif classifier in glcm_classifier:
                        if classifier == 'Glcm_MLH':
                            vettore_class = dzn_class_param[classifier_leaf[i]](glcm_train_mlh,glcm_train_cls,Glcm_Data_list_mlh,n_bands,param)
                        else:
                            vettore_class = dzn_class_param[classifier_leaf[i]](glcm_train,glcm_train_cls,Glcm_Data_list,n_bands,param)
                    elif classifier in ndsv_classifier:
                        if classifier == 'Ndsv_MLH':
                            vettore_class = dzn_class_param[classifier_leaf[i]](ndsv_train_mlh,ndsv_train_cls,Ndsv_Data_list_mlh,n_bands,param)
                        else:
                            vettore_class = dzn_class_param[classifier_leaf[i]](ndsv_train,ndsv_train_cls,Ndsv_Data_list,n_bands,param)
                    else:
                        if classifier == 'MLH':
                            vettore_class = dzn_class[classifier_leaf[i]](train_list_mlh,train_list_class,Data_list_mlh)
                        else:
                            vettore_class = dzn_class[classifier_leaf[i]](train_list,train_list_class,Data_list)
                else:
                    if classifier in morph_classifier:
                        if classifier == 'Morph_MLH':
                            vettore_class = dzn_class[classifier_leaf[i]](morph_train_mlh,morph_train_cls,Morph_Data_list_mlh)
                        else:
                            vettore_class = dzn_class[classifier_leaf[i]](morph_train,morph_train_cls,Morph_Data_list)
                    elif classifier in glcm_classifier:
                        if classifier == 'Glcm_MLH':
                            vettore_class = dzn_class[classifier_leaf[i]](glcm_train_mlh,glcm_train_cls,Glcm_Data_list_mlh)
                        else:
                            vettore_class = dzn_class[classifier_leaf[i]](glcm_train,glcm_train_cls,Glcm_Data_list)
                    elif classifier in ndsv_classifier:
                        if classifier == 'Ndsv_MLH':
                            vettore_class = dzn_class[classifier_leaf[i]](ndsv_train_mlh,ndsv_train_cls,Ndsv_Data_list_mlh)
                        else:
                            vettore_class = dzn_class[classifier_leaf[i]](ndsv_train,ndsv_train_cls,Ndsv_Data_list)
                    else:
                        if classifier == 'MLH':
                            vettore_class = dzn_class[classifier_leaf[i]](train_list_mlh,train_list_class,Data_list_mlh)
                        else:
                            vettore_class = dzn_class[classifier_leaf[i]](train_list,train_list_class,Data_list)

                vettore_class[np.where(vettore_class!=cls)]= 0
                mask = np.where(class_data != 0)
                
                for i in np.arange(j,j+split):
                    class_data_tmp[indici[0][i],indici[1][i]]= vettore_class[i-j]
                
                class_data_tmp[mask]= 0
                class_data = class_data+class_data_tmp
                
                last_class.remove(cls)
                if classifier in morph_classifier:
                    indici_cls = np.where(morph_train_cls == cls)
                elif classifier in glcm_classifier:
                    indici_cls = np.where(glcm_train_cls == cls)
                elif classifier in ndsv_classifier:
                    indici_cls = np.where(ndsv_train_cls == cls)
                else:
                    indici_cls = np.where(train_list_class == cls)
                
                train_list = np.delete(train_list,indici_cls,axis=0)
                train_list_mlh = np.delete(train_list_mlh,indici_cls,axis=0)
                train_list_class = np.delete(train_list_class,indici_cls,axis=0)
                morph_train = np.delete(morph_train,indici_cls,axis=0)
                morph_train_mlh = np.delete(morph_train_mlh,indici_cls,axis=0)
                morph_train_cls = np.delete(morph_train_cls,indici_cls,axis=0)
                glcm_train = np.delete(glcm_train,indici_cls,axis=0)
                glcm_train_mlh = np.delete(glcm_train_mlh,indici_cls,axis=0)
                glcm_train_cls = np.delete(glcm_train_cls,indici_cls,axis=0)
                if ndsv != None:
                    ndsv_train = np.delete(ndsv_train,indici_cls,axis=0)
                    ndsv_train_mlh = np.delete(ndsv_train_mlh,indici_cls,axis=0)
                    ndsv_train_cls = np.delete(ndsv_train_cls,indici_cls,axis=0)

        class_data[np.where(class_data == 0)]= last_class[0]

        ###################################################
        #RESULT

        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        if data.ndim != 2:
            outDataset = driver.Create(hdt_class_path, data.shape[2], data.shape[1], 1, gdal.GDT_Byte)
        else:
            outDataset = driver.Create(hdt_class_path, data.shape[1], data.shape[0], 1, gdal.GDT_Byte)
        outDataset.SetGeoTransform(geoTransform )
        outDataset.SetProjection(proj)
        outBand = outDataset.GetRasterBand(1)

        outBand.WriteArray(class_data, 0, 0)

        outBand = None
        outDataset = None
    
if __name__ == "__main__":
    main()
