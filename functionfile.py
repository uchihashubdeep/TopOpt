import numpy as np
import math
# import pandas as pd

def rownum(nodes):
    
    n=len(nodes)
    
    rn=np.empty([2*n,1],dtype=int)
    
    for i in range(n):
        rn[2*i]=2*nodes[i]-2
        rn[2*i+1]=2*nodes[i]-1
        
    return rn

def stiffness(C,xi,yi):
    
    sval=np.array([-1/math.sqrt(3),1/math.sqrt(3)])
    
    Ke=np.zeros([8,8])
    Bint=np.zeros([3,8])
    A=0
    
    for s in sval:
        for t in sval:
            
# =============================================================================
#             N1=(1-s)*(1-t)/4
#             N2=(1+s)*(1-t)/4
#             N3=(1+s)*(1+t)/4
#             N4=(1-s)*(1+t)/4
# =============================================================================
            
            dNs=np.array([-(1-t), 1-t, 1+t, -(1+t)])*0.25;
            dNt=np.array([-(1-s), -(1+s), 1+s, 1-s])*0.25;
            
            dxs=np.inner(dNs,xi)
            dxt=np.inner(dNt,xi)
            dys=np.inner(dNs,yi)
            dyt=np.inner(dNt,yi)
            
            J=np.array([[dxs, dxt],[dys, dyt]])
            
            detJ=np.linalg.det(J)
            
            if(detJ==0):
                detJ=10e-6
            
            dNxy=np.linalg.solve(J.T,np.vstack((dNs,dNt)))
                        
            dNx=dNxy[0,:]
            dNy=dNxy[1,:]
            
            B=np.zeros([3,8])
            
            for k in range(4):
                B[:,2*k:2*k+1+1]=np.array([[dNx[k], 0], [0, dNy[k]], [dNy[k], dNx[k]]])
                
            Ke = Ke + np.linalg.multi_dot([B.T,C,B])*detJ
            
            Bint = Bint + B*detJ
            A = A + detJ
            
    return Ke,Bint,A
            
def stress_strain(C,xi,yi,dele):
    
    sval=np.array([-1/math.sqrt(3),1/math.sqrt(3)])
    
    strain=np.empty([3,4])
    stress=np.empty([3,4])
    cnt=0
    
    for s in sval:
        for t in sval:
            
# =============================================================================
#             N1=(1-s)*(1-t)/4
#             N2=(1+s)*(1-t)/4
#             N3=(1+s)*(1+t)/4
#             N4=(1-s)*(1+t)/4
# =============================================================================
            
            dNs=np.array([-(1-t), 1-t, 1+t, -(1+t)])*0.25;
            dNt=np.array([-(1-s), -(1+s), 1+s, 1-s])*0.25;
            
            dxs=np.inner(dNs,xi)
            dxt=np.inner(dNt,xi)
            dys=np.inner(dNs,yi)
            dyt=np.inner(dNt,yi)
            
            J=np.array([[dxs, dxt],[dys, dyt]])
            
            detJ=np.linalg.det(J)
            
            dNxy=np.linalg.solve(J.T,np.vstack((dNs,dNt)))
                        
            dNx=dNxy[0,:]
            dNy=dNxy[1,:]
            
            B=np.zeros([3,8])
            
            for k in range(4):
                B[:,2*k:2*k+1+1]=np.array([[dNx[k], 0], [0, dNy[k]], [dNy[k], dNx[k]]])
                
            strain[:,cnt]=np.linalg.multi_dot([B,dele])
            cnt=cnt+1
    
    stress=np.linalg.multi_dot([C,strain])
            
    return strain,stress

def connectivity(nx,ny):
    dim=nx*ny
    nodecon=np.ones([dim,5])
    k=0
    for i in range(dim):
        nodecon[i,0]=i+1
        nodecon[i,1]=i+1+k
        nodecon[i,2]=nodecon[i,1]+1
        nodecon[i,3]=(nx+2)+(i+1)+k
        nodecon[i,4]=nodecon[i,3]-1
        if((i+1)%nx==0):
            k=k+1
            
    return nodecon.astype(int)


def coordinates(nx,ny,lx,ly):
    ix=(float)(lx/nx)
    iy=(float)(ly/ny)
    dim=(nx+1)*(ny+1)
    nodecoord=np.ones([dim,3])
    k=0
    n=0
    for i in range(dim):
        
        nodecoord[i,0]=i+1
        nodecoord[i,2]=(float)(k*iy-(ly/2))
        nodecoord[i,1]=float(ix*n)
        n=n+1
        if((i+1)%(nx+1)==0):
            k=k+1
            n=0
    return nodecoord



def fixed(nx,ny):
    numnode=(nx+1)*(ny+1)
    fixed_nodes=np.arange(1,numnode,nx+1)
    return fixed_nodes


def Load(x,y):
    return (x+1)*np.ceil(((y+1)/2))


def dimension(nodecoord):  #to find number of rows and columns
    row=0
    column=0
    size=nodecoord.shape[0]
    for i in range(size):
        if(nodecoord[i,1]==0):
            row=row+1
        if(nodecoord[i,2]==-0.5):
            column=column+1
    return row-1,column-1

