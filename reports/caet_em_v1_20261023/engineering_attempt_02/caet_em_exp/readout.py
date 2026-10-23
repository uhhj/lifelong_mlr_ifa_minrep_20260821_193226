import torch
def readout(transformer,x):
 assert x.ndim==4 and 1<=x.shape[1]<=8 and x.shape[2:]==(4,512),x.shape
 b,t,m,e=x.shape;y=transformer(x.reshape(b,t*m,e)).reshape(b,t,m,e)
 return y[:,-1,3],y[:,-1,0]
