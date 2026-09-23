import unittest,torch
from .project import project,orthonormal_random
class ProjectorTest(unittest.TestCase):
 def test_rotation_and_projection(self):
  U=orthonormal_random(15,4,73,'cpu');R=orthonormal_random(4,4,74,'cpu');v=torch.arange(15,dtype=torch.float64)
  p=project(U,v);self.assertTrue(torch.allclose(p,project(U@R,v),atol=1e-12,rtol=1e-12))
  self.assertTrue(torch.allclose(project(U,p),p,atol=1e-12,rtol=1e-12))
  self.assertAlmostEqual(float(v@p),float(p@p),places=10)
  scaled=U*torch.tensor([.9,1.1,1.2,.8],dtype=torch.float64)
  self.assertTrue(torch.allclose(p,project(scaled,v),atol=1e-12,rtol=1e-12))
if __name__=='__main__':unittest.main()
