__version__ = '1.0'

import SingleCloneDistance as d
import numpy as np

def scatterPoints(randomPoints, translations, precision, center, genNum, L3):
    if genNum == 3:
        L_scatter = [(randomPoints[i][0] * translations[0] + randomPoints[i][1] * translations[1] - randomPoints[i][2] * translations[2]) for i in range(precision)]
    else:
        L_scatter = [(randomPoints[i][0] * translations[0] + randomPoints[i][1] * translations[1] + (randomPoints[i][2]-0.5)*np.array([0,0,2*L3])) for i in range(precision)]
    return(L_scatter)


def samplePoints(Manifold, angles, precision, L_Scale):
    
    if Manifold in {'E1','E2','E3','E4','E5','E6'}:
        genNum = 3
    else:
        genNum = 2
    
    randomPoints = np.random.rand(precision, 3)
    
    if genNum == 3:
        M, translations, pureTranslations, E1Dict, center, x0 = d.manifolds.construct3Generators(Manifold, L_Scale, angles)
        L_scatter = scatterPoints(randomPoints, pureTranslations, precision, center, genNum, L_Scale[2])
        pos = L_scatter - 0.5*((pureTranslations[0] + pureTranslations[1] - pureTranslations[2]))
    else:
        M, translations, pureTranslations, E1Dict, center, x0 = d.manifolds.construct2Generators(Manifold, L_Scale, angles)
        L_scatter = scatterPoints(randomPoints, pureTranslations, precision, center, genNum, L_Scale[2])
        pos = L_scatter - 0.5*((pureTranslations[0] + pureTranslations[1]))
    
    count = 0
    excludedPoints = []
    allowedPoints = []

    for k in range(precision):
        
        
        dist = d.sampleTopol(Manifold, L_Scale, pos[k], angles)
        
        if dist < 1:
            count +=1
            excludedPoints.append(pos[k])
        else:
            allowedPoints.append(pos[k])


    percents = 1 - (count/precision)
    
    excludedPoints = np.array(excludedPoints).tolist()
    allowedPoints = np.array(allowedPoints).tolist()

    L_x = [allowedPoints[i][0] for i in range(len(allowedPoints))]
    L_y = [allowedPoints[i][1] for i in range(len(allowedPoints))]
    L_z = [allowedPoints[i][2] for i in range(len(allowedPoints))]
    excludedPoints_x = [excludedPoints[i][0] for i in range(len(excludedPoints))]
    excludedPoints_y = [excludedPoints[i][1] for i in range(len(excludedPoints))]
    excludedPoints_z = [excludedPoints[i][2] for i in range(len(excludedPoints))]
            
    return(percents, [excludedPoints_x, excludedPoints_y, excludedPoints_z], [L_x, L_y, L_z])





