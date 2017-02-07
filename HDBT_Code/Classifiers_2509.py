# -*- coding: cp1252 -*-

from AccuracyAssesment_PRIN_2509 import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, grid_search, preprocessing, metrics


def KNNeighbors(train_list,train_list_class,GT_list,GT_list_class):

    knn = KNeighborsClassifier()
    knn.fit(train_list, train_list_class)
    result = knn.predict(GT_list)
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)

    return Conf_Matr

def KNNeighbors_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    knn = KNeighborsClassifier()
    knn.fit(train_list[:,temp_feat], train_list_class)
    result = knn.predict(GT_list[:,temp_feat])
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)
    temp = list(Conf_Matr)
    temp.append(list_param)
    Conf_Matr_FIN = tuple(temp)
    
    return Conf_Matr_FIN

def KNNeighbors_class(train_list,train_list_class,Data_list):

    knn = KNeighborsClassifier()
    knn.fit(train_list, train_list_class)
    result = knn.predict(Data_list)

    return result

def KNNeighbors_class_param(train_list,train_list_class,Data_list,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    knn = KNeighborsClassifier()
    knn.fit(train_list[:,temp_feat], train_list_class)
    result = knn.predict(Data_list[:,temp_feat])

    return result

def RForest(train_list,train_list_class,GT_list,GT_list_class):

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_list, train_list_class)
    result = clf.predict(GT_list)
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)
    
    return Conf_Matr


def RForest_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_list[:,temp_feat], train_list_class)
    result = clf.predict(GT_list[:,temp_feat])
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)
    temp = list(Conf_Matr)
    temp.append(list_param)
    Conf_Matr_FIN = tuple(temp)
    
    return Conf_Matr_FIN

def RForest_class(train_list,train_list_class,Data_list):

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_list, train_list_class)
    result = clf.predict(Data_list)

    return result

def RForest_class_param(train_list,train_list_class,Data_list,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train_list[:,temp_feat], train_list_class)
    result = clf.predict(Data_list[:,temp_feat])

    return result


def SVMachine(train_list,train_list_class,GT_list,GT_list_class):

    C_range = 10.0 ** np.arange(0, 3)
    gamma_range = 10.0 ** np.arange(-2, 1)
    param_grid = {"gamma":gamma_range.tolist(), "C":C_range.tolist()}
    svr = svm.SVC(kernel='rbf',cache_size=1000)
    clfopt = grid_search.GridSearchCV(svr,param_grid)
    clfopt.fit(train_list, train_list_class)
    
    #Define a SVM using the best parameters C and gamma
    clf = svm.SVC(gamma = clfopt.best_estimator_.gamma, C = clfopt.best_estimator_.C,kernel='rbf',cache_size=1000)
    clf.fit(train_list, train_list_class)
    result = clf.predict(GT_list)
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)

    return Conf_Matr

def SVMachine_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    C_range = 10.0 ** np.arange(0, 3)
    gamma_range = 10.0 ** np.arange(-2, 1)
    param_grid = {"gamma":gamma_range.tolist(), "C":C_range.tolist()}
    svr = svm.SVC(kernel='rbf',cache_size=1000)
    clfopt = grid_search.GridSearchCV(svr,param_grid)
    clfopt.fit(train_list[:,temp_feat], train_list_class)
    
    #Define a SVM using the best parameters C and gamma
    clf = svm.SVC(gamma = clfopt.best_estimator_.gamma, C = clfopt.best_estimator_.C,kernel='rbf',cache_size=1000)
    clf.fit(train_list[:,temp_feat], train_list_class)
    result = clf.predict(GT_list[:,temp_feat])
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)
    temp = list(Conf_Matr)
    temp.append(list_param)
    Conf_Matr_FIN = tuple(temp)

    return Conf_Matr_FIN

def SVMachine_class(train_list,train_list_class,Data_list):

    C_range = 10.0 ** np.arange(0, 3)
    gamma_range = 10.0 ** np.arange(-2, 1)
    param_grid = {"gamma":gamma_range.tolist(), "C":C_range.tolist()}
    svr = svm.SVC(kernel='rbf',cache_size=1000)
    clfopt = grid_search.GridSearchCV(svr,param_grid)
    clfopt.fit(train_list, train_list_class)

    #Define a SVM using the best parameters C and gamma
    clf = svm.SVC(gamma = clfopt.best_estimator_.gamma, C = clfopt.best_estimator_.C,kernel='rbf',cache_size=1000)
    clf.fit(train_list, train_list_class)
    result = clf.predict(Data_list)
    
    return result

def SVMachine_class_param(train_list,train_list_class,Data_list,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    C_range = 10.0 ** np.arange(0, 3)
    gamma_range = 10.0 ** np.arange(-2, 1)
    param_grid = {"gamma":gamma_range.tolist(), "C":C_range.tolist()}
    svr = svm.SVC(kernel='rbf',cache_size=1000)
    clfopt = grid_search.GridSearchCV(svr,param_grid)
    clfopt.fit(train_list[:,temp_feat], train_list_class)

    #Define a SVM using the best parameters C and gamma
    clf = svm.SVC(gamma = clfopt.best_estimator_.gamma, C = clfopt.best_estimator_.C,kernel='rbf',cache_size=1000)
    clf.fit(train_list[:,temp_feat], train_list_class)
    result = clf.predict(Data_list[:,temp_feat])
    
    return result

def NB_MLHood(train_list,train_list_class,GT_list,GT_list_class):

    gp = GaussianNB()
    gp.fit(train_list,train_list_class) 
    result = gp.predict(GT_list)
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)

    return Conf_Matr

def NB_MLHood_param(train_list,train_list_class,GT_list,GT_list_class,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)
    gp = GaussianNB()
    gp.fit(train_list[:,temp_feat],train_list_class) 
    result = gp.predict(GT_list[:,temp_feat])
    Conf_Matr = Conf_Matr_2arr(result,GT_list_class)
    temp = list(Conf_Matr)
    temp.append(list_param)
    Conf_Matr_FIN = tuple(temp)
    
    return Conf_Matr_FIN

def NB_MLHood_class(train_list,train_list_class,Data_list):

    gp = GaussianNB()
    gp.fit(train_list,train_list_class) 
    result = gp.predict(Data_list)

    return result

def NB_MLHood_class_param(train_list,train_list_class,Data_list,n_bands,list_param):

    base = np.arange(n_bands)
    temp_feat = np.concatenate((base,list_param)).astype(int)    
    gp = GaussianNB()
    gp.fit(train_list[:,temp_feat],train_list_class) 
    result = gp.predict(Data_list[:,temp_feat])

    return result
