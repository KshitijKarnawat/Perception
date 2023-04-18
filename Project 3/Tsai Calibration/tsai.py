#!/usr/bin/env python3

"""
 * @copyright Copyright (c) 2023
 * @file tsai.py
 * @author Kshitij Karnawat (kshitij@umd.edu)
 * @brief Question 1 for Porject 3
 * @version 0.1
 * @date 2023-04-17
 * 
 * 
"""
import numpy as np
import scipy as sp


def reprojection(u,v,X,Y,Z):
    """
    Calculates the reprojection error

    Input:  u(list): x coordinates of image
            v(list): y coordinates of image
            X(list): x coordinates of world
            Y(list): y coordinates of world
            Z(list): z coordinates of world
    
    Returns: None
    """

    A = np.zeros((2,12), dtype=np.int8)

    for i in range(len(X)):
        a_i = np.array([[X[i], Y[i], Z[i], 1, 0, 0, 0, 0, (-u[i] * X[i]), (-u[i] * Y[i]), (-u[i] * Z[i]), (-u[i])],
                        [0, 0, 0, 0, X[i], Y[i], Z[i], 1, (-v[i] * X[i]), (-v[i] * Y[i]), (-v[i] * Z[i]), (-v[i])]])
        A = np.vstack((A,a_i))
    A = A[2:]

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T,A))
    eigenvalues_idx = np.argmin(eigenvalues)          
    projection_matrix = eigenvectors[:,eigenvalues_idx]
    projection_matrix = projection_matrix.reshape(3,4)
    print("Projection Matrix =", projection_matrix)

    M = projection_matrix[:,:3]

    intrinsic_matrix, rotation = sp.linalg.rq(M)
    intrinsic_matrix = intrinsic_matrix/intrinsic_matrix[-1,-1]
    print("Intrinsic Matrix =", intrinsic_matrix)
    print("Rotation =", rotation)

    translation = np.dot(np.linalg.inv(intrinsic_matrix), projection_matrix[:,-1])
    print("Translation =", translation)

    error = 0
    for i in range(len(X)):
        reprojection = np.dot(projection_matrix, np.array([X[i], Y[i], Z[i], 1]))
        reprojection = reprojection/reprojection[-1]
        reprojection_error = np.linalg.norm(reprojection - np.array([u[i], v[i] ,1]))
        print("Reprojection Error for point [",u[i],v[i],"] =",reprojection_error)
        error += reprojection_error
    print("Mean Reprojection error =", error/len(X))

#Pixel Coordinates
u = np.array([757, 758, 758, 759, 1190, 329, 1204, 340])
v = np.array([213, 415, 686, 966, 172, 1041, 850, 159])

#World Coordinates
X = np.array([0, 0, 0, 0, 7, 0, 7, 0])
Y = np.array([0, 3, 7, 11, 1, 11, 9, 1])
Z = np.array([0, 0, 0, 0, 0, 7, 0, 7])

reprojection(u, v, X, Y, Z)

X = np.delete(X, [1,2])
Y = np.delete(Y, [1,2])
Z = np.delete(Z, [1,2])
u = np.delete(u, [1,2])
v = np.delete(v, [1,2])

reprojection(u, v, X, Y, Z)