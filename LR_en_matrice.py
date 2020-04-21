# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:21:05 2020

@author: LENOVO
"""

import numpy as np 
import matplotlib.pyplot as plt 
from  sklearn.datasets import make_regression


x,y = make_regression(n_samples=100 , n_features=1 , noise=10)

y=y.reshape(len(y),1)

X = np.hstack((x,np.ones(x.shape)))

theta = np.random.randn(2,1)



m=len(x)
#definition du modele
def model(X,theta):
    return X.dot(theta)

#plt.scatter(x,model(X,theta) , c='r')
#plt.scatter(x,y)


#definition de la fonction cout 
def cost_function(X , theta , y ):
    return 1/(2*m) * np.sum((model(X,theta)-y)**2)

#definition du gradient 
    
def gradient(X , theta , y ):
    return 1/m * X.T.dot(model(X,theta)-y)

#definition de l'algorithme d'apprentissage  (gradient descent )
def gradient_descent(X,theta , y , nb_ite = 1000 , lr=0.001):
    cost = np.zeros(nb_ite);
    for i in range(nb_ite):
        theta=theta-lr*gradient(X , theta , y )
        cost[i]=cost_function(X , theta , y )
        
    return theta , cost
        
nb_ite = 400 
lr=0.01
theta_final , cost = gradient_descent(X,theta , y , nb_ite , lr)


plt.plot(x,model(X,theta_final) , c='r')
plt.scatter(x,y)

plt.plot(range(nb_ite),cost)


def coef_det(y, pred):
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    
    return 1-u/v

coef = coef_det(y, model(X,theta_final))  
print(coef)
    
    