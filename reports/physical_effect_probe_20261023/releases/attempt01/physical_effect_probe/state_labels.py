"""World-coordinate recorded-state labels. No simulator, action replay or model."""
import xml.etree.ElementTree as ET
import numpy as np

TARGET={6:'butter',7:'milk'}
def position(states,xml,obj):
 tree=ET.fromstring(xml);world=tree.find('worldbody');parents={c:p for p in tree.iter() for c in p};q=0
 for n in world.iter():
  if n.tag not in ('joint','freejoint'):continue
  kind='free' if n.tag=='freejoint' else n.get('type','hinge')
  if n.get('name')==obj+'_1_joint0':
   assert kind=='free' and parents[parents[n]] is world
   assert np.allclose(np.linalg.norm(states[:,1+q+3:1+q+7],axis=1),1,atol=1e-6,rtol=0)
   return states[:,1+q:1+q+3]
  q+={'free':7,'ball':4,'hinge':1,'slide':1}[kind]
 raise AssertionError(('unverified object joint',obj))
def recorded(g,task):
 s=g['states'][:];j=g['obs/joint_states'][:];gr=g['obs/gripper_states'][:];eef=g['obs/ee_pos'][:]
 assert len(s)==len(eef)==len(j)==len(gr)
 assert np.array_equal(s[1:,1:8],j[:-1]) and np.array_equal(s[1:,8:10],gr[:-1])
 xml=g.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else xml
 obj=position(s,xml,TARGET[task]);goal=position(s,xml,'basket');gap=np.abs(gr[:,0]-gr[:,1])
 # q[u] is current state at observation u (recorded s[u+1]); final unavailable row remains masked.
 q=np.full((len(s),7),np.nan);q[:-1]=np.concatenate([obj[1:]-eef[:-1],obj[1:]-goal[1:],gap[:-1,None]],axis=1)
 return q,obj,eef,gap,xml
