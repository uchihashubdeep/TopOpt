import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from functionfile import *

def planestress_topopt(nodecoord,nodecon,fixed_nodes,E,nu):
    
# =============================================================================
#     E - (2,) vector of Young's Modulus
#     nu - (2,) vector of Poisson's Ratio
#     fixed_nodes - (n,) vector of fixed nodes -> Specify as int
#     nodecoord - Nodal Coordinates
#     nodecon - Nodal connectivity with last column as material number - int datatype
# =============================================================================
    
    numnode = np.size(nodecoord,0)
    numele = np.size(nodecon,0)


    knowndof=rownum(fixed_nodes)

    activedof=np.setdiff1d(np.arange(0,2*numnode),knowndof)

    xmax=np.squeeze(np.where(nodecoord[:,1]==np.amax(nodecoord[:,1])))
    xmin=np.squeeze(np.where(nodecoord[:,1]==np.amin(nodecoord[:,1])))
    ymax=np.squeeze(np.where(nodecoord[:,2]==np.amax(nodecoord[:,2])))
    ymin=np.squeeze(np.where(nodecoord[:,2]==np.amin(nodecoord[:,2])))
    centerline=np.squeeze(np.where(nodecoord[:,2]==0))

    E1=E[0]
    E2=E[1]
    
    nu1=nu[0]
    nu2=nu[1]
    
    C1=np.array([[1, nu1, 0], [nu1, 1, 0], [0, 0, (1-nu1)/2]])*(E1/(1-nu1**2))
    C2=np.array([[1, nu2, 0], [nu2, 1, 0], [0, 0, (1-nu2)/2]])*(E2/(1-nu2**2))
    
    kg=np.zeros((numnode*2,numnode*2))
    fg=np.zeros((numnode*2,1))
    
    for i in range(numele):
        nodes = nodecon[i,1:5]
        xi = nodecoord[nodes-1,1]
        yi = nodecoord[nodes-1,2]
    

        rn = np.empty([8,1])
        rn = rownum(nodes)
        
        if (nodecon[i,5]==1):
            Ke,Bint,A = stiffness(C1,xi,yi)
        else:
            Ke,Bint,A = stiffness(C2,xi,yi)
            
        Ke=Ke*0.05
        kg[rn.T,rn] = kg[rn.T,rn] + Ke
    
    fg[2*189-1] = -1
    
    d=np.zeros([2*numnode])
        
    d[activedof]=np.linalg.solve(kg[activedof[:,None],activedof[:,None].T],fg[activedof]).reshape(-1)
    
    objfun = np.linalg.multi_dot([d.T,kg,d])
    
    return objfun
