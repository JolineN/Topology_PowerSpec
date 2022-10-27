#!/usr/bin/env python
# coding: utf-8

# ### Finds distance to closest clone for a given Fundamental Domain

__version__ = '1.0'

get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv, pinv
import itertools as it


# ## Distance Functions

def sampleTopol(Manifold, L_Scale, pos, angles):
    """
    Holds the function for performing a sampling of a fundamental domain without the need to find the closest clone location.
    """
    M, translations, pureTranslations, E1Dict, translationList, genNum, x0 = constructions(Manifold, L_Scale, angles)
    
    dist = E_general_topol(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList)
    return (dist)


def distance_to_CC(Manifold, L_Scale, pos, angles):

    """
    Holds the function for running the distance and clone location code.
    """

    M, translations, pureTranslations, E1Dict, translationList, genNum, x0 = constructions(Manifold, L_Scale, angles)

    if (Manifold in {'E1', 'E11'}):
        closestClone, dist, genApplied = E1_Associated_Trans(pureTranslations, pos)

    else:
        closestClone, dist, genApplied = E_general(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList)

    return(closestClone, dist, genApplied)


# ## Code for performing the distance calculations

def constructions(Manifold, L_Scale, angles):
    
    """
    Constructs the foundational values (generators and their individual pieces [translations, matricies, etc.]) 
    for the problem (based on given values and chosen Manifold)
    """
    
    _3Gen = {'E1','E2','E3','E4','E5','E6'}
    _2Gen = {'E11','E12','E16'}
    
    if (Manifold in _3Gen):
        genNum = 3
        M, translations, pureTranslations, E1Dict, center, x0 = manifolds.construct3Generators(Manifold, L_Scale, angles)
        
        if center == True:
            translationList = findAllTranslationsCenter(pureTranslations, genNum)
        else:
            translationList = findAllTranslationsCorner(pureTranslations)
        
    elif (Manifold in _2Gen):
        genNum = 2
        M, translations, pureTranslations, E1Dict, center, x0 = manifolds.construct2Generators(Manifold, L_Scale, angles)
        translationList = findAllTranslationsCenter(pureTranslations, genNum)
        
    return (M, translations, pureTranslations, E1Dict, translationList, genNum, x0)



def E1_Associated_Trans(pureTranslations, pos):
    gens = ['g1','g2','g3']
    
    translationList = pureTranslations
    nearestClone = [distance.euclidean(pureTranslations[i], x0) for i in range(len(pureTranslations))]
    _minNear = min(nearestClone)
    indexOfClone = nearestClone.index(_minNear)
    shortestTrans = gens[indexOfClone]
    
    return(pureTranslations[indexOfClone] + pos, _minNear, shortestTrans)



def E_general(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList):
    
    clonePositions, genApplied = findClones(pos, x0, M, translations, E1Dict, genNum)
    translatedClonePos = [translateClones(clonePositions[i], translationList) for i in range(len(clonePositions))]
    nearestFromLayer = [distances(translatedClonePos[i], pos, x0) for i in range(len(translatedClonePos))]
    closestClone = findClosestClone(nearestFromLayer, pureTranslations, x0, pos)
    generatorCombo = findGeneratorCombo(closestClone[0], clonePositions, pureTranslations, pos, E1Dict, Manifold, genApplied, genNum)
    return (closestClone[0], closestClone[1], generatorCombo)


def E_general_topol(Manifold, pos, x0, M, translations, pureTranslations, E1Dict, genNum, translationList):
    
    clonePositions, genApplied = findClones(pos, x0, M, translations, E1Dict, genNum)
    translatedClonePos = [translateClones(clonePositions[i], translationList) for i in range(len(clonePositions))]
    nearestFromLayer = [distances(translatedClonePos[i], pos, x0) for i in range(len(translatedClonePos))]
    closestClone = findClosestClone(nearestFromLayer, pureTranslations, x0, pos)
    
    return (closestClone[1])


# Finds all possible translations in the positive direction by creating combinations of pure translations

def findAllTranslationsCorner(pureTranslations):
    
    """Determines all the combinations of pure translations for manifolds with the origin at the corner of the fundamental domain."""
    
    _trans1 = [list(it.combinations_with_replacement(pureTranslations, i)) for i in range(len(pureTranslations) + 2)]
    _trans2 = [[(np.add.reduce(_trans1[i][j])) for j in range(len(_trans1[i]))] for i in range(len(_trans1))]
    
    
    _trans2.append([pureTranslations[0] - pureTranslations[1]])
    _trans2.append([pureTranslations[1] - pureTranslations[0]])
    
    _trans2[0] = [[0,0,0]]
    
    transUpPlane = list(it.chain.from_iterable(_trans2))
    allnewTrans = np.array((np.unique(transUpPlane, axis = 0)))
    
    return(allnewTrans)



def findAllTranslationsCenter(pureTranslations, genNum):
    """Determines all the combinations of pure translations for manifolds with the origin at the center of the associated E1."""
    layerTrans = [pureTranslations[0],    pureTranslations[1],       -pureTranslations[0],   -pureTranslations[1],
               -2*pureTranslations[0], -2*pureTranslations[1],      2*pureTranslations[0],  2*pureTranslations[1],
                  pureTranslations[0] + pureTranslations[1],        pureTranslations[0] - pureTranslations[1], 
                 -pureTranslations[0] + pureTranslations[1],       -pureTranslations[0] - pureTranslations[1]] 
                 
    if genNum == 3:
        allnewTrans = np.concatenate([layerTrans, layerTrans + pureTranslations[2], [pureTranslations[2]]])
        return(allnewTrans)
    else:
        return(layerTrans)


# ### Finds all the clones up to associated E1
# Takes the original position and applies combinations of the generators (up to E1 for each generator) and returns a list of clones


def findClones(pos, x0, M, translations, E1Dict, genNum):
    """The first loop (clonePos) determines the arrangements generators need to be applied in order to fully represent each of the clones (up to the associated E1).
    For example, an element of clonePos may look like [1,1,2] which means applying g1 . g2 . g3 . g3 to the initial position.
    
    The second loop (fullCloneList) determines the new position after applying the combination of generators described by the corrosponding element in clonePos. 
    """
    clonePos = []
    
    for i in range(E1Dict[0]):
        for j in range(E1Dict[1]):
            if(genNum == 3):
                for k in range(E1Dict[2]):
                    clonePos.append([i,j,k])
            else:
                clonePos.append([i,j])

    fullCloneList = []
    
    for i in range(len(clonePos)):
        if not (all(x == 0 for x in clonePos[i])):
            _x = pos
            for j in range(len(clonePos[i])):
                for k in range(clonePos[i][j]):
                    _x = generatorPos(_x, x0, M[j], translations[j])
            fullCloneList.append(_x)
    
    if not fullCloneList:
        return(pos, clonePos)
    else:
        return(fullCloneList, clonePos)


# #### Applies a generator to the input point


def generatorPos(x,x0, M, translations):
    """Application of a generator to an initial point, x"""
    
    x_out = M.dot(x-x0) + translations + x0
    
    return(x_out)


# #### Translates a clones position for each of the allowed translations



def translateClones(clonePos, translations):
    """Translates the clone by all the combinations of pure translations in the "translations" list. The pure translations are determined based on the
    parameters and the given manifold."""
    
    translatedClonePos = [(clonePos + translations[i]) for i in range(len(translations))]
    
    return(translatedClonePos)


# #### Finds the distance between each translated clone and the original position


def distances(clonePos, pos, x0):
    
    """Determines the distance between the inital position and each of the clones in a single "layer". That is, after applying a generator (or combination
    of generators) and then all the pre-determined pure translations to that clone, which of these new clones is closest to the original position.
    For example, E3 has 3 unique clones (g3, g3^2, and g3^3) and so has 3 unique layers. This function returns the closest clone (to the original position)
    in each of these layers"""
    
    _TransDist = [distance.euclidean(pos, clonePos[i]) for i in range(len(clonePos))]
    min_TransDist = min(_TransDist)
    closestClonePos = clonePos[_TransDist.index(min_TransDist)]
    
    return(closestClonePos, min_TransDist)


# ### Finds the closest clone from all translated clones
# Takes an input of all the translated clones and compares their positions to the original position. Finds the minimum of pure translations. Returns the minimum of these two values


def findClosestClone(generatedClones, pureTrans, x0, pos):
    
    """Determines the closest clone from the list of closest clones. For example, in E3 we have 3 layers (described above), each with their own closest clone.
    This function returns the closest of these 3 clones."""
    
    _TranslateClone = [[(pureTrans[x] + pos) , distance.euclidean(x0,pureTrans[x])] for x in range(len(pureTrans))] 
    
    _closestTranslatedClone = min(_TranslateClone, key = lambda x: x[1] if (x[1] > 10e-12) else np.nan)
    _closestGeneratedClone = min(generatedClones, key = lambda x: x[1] if (x[1]> 10e-12) else np.nan, default = _closestTranslatedClone)

    return(min((_closestGeneratedClone, _closestTranslatedClone), key = lambda x: x[1]))


# #### Finds Combination of Generators to Produce closest Clone
# TLDR: Takes the input of the closest clones position, all the generated clones (up to associated E1), pure translations, etc. and determines what combinatinos of generators will produce that closest clone position.
# 
# Detailed Description: Takes the list of all clones (up to associated E1) and subtracts the closest clone position. This should produce a set of new points (without a relevant physical interpretation) that are linear combinations of the three pure translation vectors. One of those linear combinations will be a set of integers: the one with the corrosponding, non-trivial, generator. Next, this determines which new point is that vector of only integers (and what those integers are), determines what the original clone was, and combines the original clones generator, and the linear combination of translations to return the full list of generators.


def findGeneratorCombo(pos, clones, translations, origPos, E1Dict, Manifold, genApplied, genNum):
    """
    Description: Finds the series of generators that produces the closest clone. 
    
    Method: Determines which clone from the set produced by the findClones() function gave the closest clone. 
    Finds the linear combination of pure translations applied after. Returns a description of non-trivial generator and pure translations.
    Example would be 'Apply g3 once, pure translations of g1, g2^2'
    """
    
    gens = [f'g1', f'g2', f'g3']
    
    if Manifold in {'E1', 'E11'}:
        _x = [clones - pos]
    else:
        _x = clones - pos
    _z = np.insert(_x, 0,[origPos - pos], axis = 0)
    
    if (genNum == 3):
        n_list = np.around([((_z[i]).dot(inv(translations))) for i in range(len(_z))], decimals = 8)
    else:
        n_list = np.around([((_z[i]).dot(pinv(translations))) for i in range(len(_z))], decimals = 8)
        
    
    test = 0
    maxZeroes = 0

    for i in range(len(n_list)):
        count = 0
        isZero = 0
        for j in range(len(n_list[i])):
            if ((n_list[i][j]).is_integer()) == True:
                count +=1
                if (n_list[i][j]) == 0:
                    isZero +=1
                    
        if count == len(n_list[i]):
            if isZero == len(n_list[i]):
                transArray, case = n_list[i], i
                break
            else:
                transArray, case = n_list[i], i
                maxZeroes = isZero
        
    initialClone = [genApplied[case][i]*gens[i] for i in range(len(genApplied[case]))]
    genList = [k for k in initialClone if k] +[f'pure translations: '] + [(gens[i], -transArray[i]) for i in range(len(genApplied[0]))]
        
    return(genList)


# #### Class containing all the manifolds and relevant quantities


class manifolds:
    """
    Class containing the manifold constructions, divided between 2 and 3 generator manifolds
    """
        
    def construct3Generators(Manifold, L_Scale, angles):
        L1, L2, L3 = L_Scale[0], L_Scale[1], L_Scale[2]
        
        M = []
        
        if (Manifold == "E1"):
            M1 = M2 = M3 = np.identity(3)
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 1
            T1 = TA1 = L1 * np.array([1,0,0])
            T2 = TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = TB  = L3 * np.array([np.cos(angles[1])*np.cos(angles[2]), np.cos(angles[1])*np.sin(angles[2]), np.sin(angles[1])])
            center = False
        
        
        elif (Manifold == "E2"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 2
            M1 = M2 = np.identity(3)
            M3 = np.diag([-1,-1,1])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            TB  = np.array([0, 0, L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = np.array([0, 0, 2*L3])
            center = True
            
            
        elif (Manifold == "E3"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 4
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[0,  1,  0],
                                 [-1, 0,  0],
                                 [0,  0,  1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            T3 = np.array([0,0,4*L3])
            
            center = True
            
            if (L1 != L2 or angles[0] != np.pi/2):
                raise ValueError("Restrictions on E3: L1=L2 and alpha = pi/2")
        
        
        elif (Manifold == "E4"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 3
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[-1/2,           np.sqrt(3)/2,  0],
                                 [-np.sqrt(3)/2, -1/2,           0],
                                 [0,              0,             1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            T3 = L3 * np.array([0, 0, 3*np.sin(angles[1])])
            T3 = np.array([0,0,3*L3])
            
            center = True
            
            if (L1 != L2):
                raise ValueError("Restrictions on E4: L1=L2")
           
        
        elif (Manifold == "E5"):
            E1_g1 = 1
            E1_g2 = 1
            E1_g3 = 6
            
            M1 = M2 = MA = np.identity(3)
            M3  = MB = np.array([[1/2,           np.sqrt(3)/2,   0],
                                 [-np.sqrt(3)/2, 1/2,            0],
                                 [0,              0,             1]])
            
            TA1 = L1 * np.array([1,0,0])
            TA2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            TB = np.array([0,0,L3])
            
            T1 = L1 * np.array([1,0,0])
            T2 = L2 * np.array([-1/2, np.sqrt(3)/2,0])
            T3 = np.array([0,0,6*L3])
            
            center = True
            
            if (L1 != L2):
                raise ValueError("Restrictions on E5: L1=L2")
                
                
        elif (Manifold == "E6"):
            LCx, LAy, LBz = L_Scale[0], L_Scale[1], L_Scale[2]
            
            E1_g1 = 2
            E1_g2 = 2
            E1_g3 = 2
            
            M1 = np.diag(([1,  -1, -1]))
            M2 = np.diag(([-1,  1, -1]))
            M3 = np.diag(([-1, -1,  1]))
            
            LAx = LCx
            LBy = LAy
            LCz = LBz
            
            TA1 = np.array([LAx, LAy,    0])
            TA2 = np.array([0,   LBy,  LBz])
            TB  = np.array([LCx,   0,  LCz])
            
            T1 = 2*LAx * np.array([1,0,0])
            T2 = 2*LBy * np.array([0,1,0])
            T3 = 2*LCz * np.array([0,0,1])
            
            center = True          
                
        
            
        translations = np.around(np.array([TA1, TA2, TB]), decimals = 5)
        pureTranslations = np.around(np.array([T1, T2, -T3]), decimals = 5)
        associatedE1Dict = np.array([E1_g1, E1_g2, E1_g3])
        M = [M1, M2, M3]
        #x0 = 0.5*np.sum(translations, axis =0)
        x0 = np.array([0,0,0.])
        
        return(M, translations, pureTranslations, associatedE1Dict, center, x0)
            


    def construct2Generators(Manifold, L_Scale, angles):
        #L1, L2, L3x, L3y, L3z = L_Scale[0], L_Scale[1], L_Scale[2], L_Scale[3], L_Scale[4]
        L1, L2 = L_Scale[0], L_Scale[1]
        
        if (Manifold == 'E11'):
            E1_g1 = 1
            E1_g2 = 1
            
            M1 = M2 = np.identity(3)
            TA1 = T1 = L1 * np.array([1,0,0])
            TA2 = T2 = L2 * np.array([np.cos(angles[0]),np.sin(angles[0]),0])
            
            center = False
            
        
        elif (Manifold == 'E12'):
            E1_g1 = 1
            E1_g2 = 2
            
            M1 = np.identity(3)
            M2 = np.diag([-1, 1, -1])
            
            TA1 = T1 = L1 * np.array([np.cos(angles[0]),0, np.sin(angles[0])])
            TA2 = np.array([0,L2,0])
            
            T2 = np.array([0,2*L2, 0])
            
            center = True
            
            
        translations = np.around(np.array([TA1, TA2]), decimals = 5)
        pureTranslations = np.around(np.array([T1, T2]), decimals = 5)
        associatedE1Dict = np.array([E1_g1, E1_g2])
        M = [M1, M2]
        x0 = np.array([0,0,0.])
        
        return(M,translations, pureTranslations, associatedE1Dict, center, x0)
