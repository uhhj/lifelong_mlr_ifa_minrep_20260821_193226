import numpy as np
SCHEMA='pep_world_delta_object_delta_eef_minus_object_delta_aperture_v1'
def effects(obj,eef,gap):
 n=len(obj);e=np.full((n,7),np.nan)
 # obs[u+1] eef and gap correspond to s[u+2].
 e[:-2]=np.concatenate([obj[2:]-obj[1:-1],(eef[1:-1]-obj[2:])-(eef[:-2]-obj[1:-1]),(gap[1:-1]-gap[:-2])[:,None]],axis=1)
 return e
