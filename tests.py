
import unittest
import numpy
import numpy.random

from transformations import Affine

EPSILON = 1e-9

class TestAffine(unittest.TestCase):
    
    def randomVector(self):
        return numpy.random.normal(size=(3,))
    
    def randomMatrix(self):
        return numpy.random.normal(size=(3,3))
    
    def assertVecsEqual( self, a, b ):
        a = numpy.ravel(a)
        b = numpy.ravel(b)
        self.assertTrue( numpy.linalg.norm(a-b) < EPSILON )
    
    def test_identity(self):
        vec = self.randomVector()
        t = Affine.identity()
        
        self.assertVecsEqual( vec, t(vec) )
        
        self.assertTrue( t.is_identity() )
        self.assertFalse( Affine(translation=(1,2,3)).is_identity() )
        self.assertFalse( Affine(rotation_axis='y', rotation_deg=10).is_identity() )
        
        rotCcw = Affine(rotation_axis=(1,0,0), rotation_deg=90)
        rotCw = Affine(rotation_axis='x', rotation_deg=-90)
        self.assertTrue(rotCcw(rotCw).is_identity())
    
    def test_translation(self):
        
        t = Affine(translation=(1,2,3))
        vec = (0,-1,-3)
        r = t(vec)
        
        self.assertVecsEqual( r, (1,1,0) )
    
    def test_inverse(self):
        
        t = Affine(linear=self.randomMatrix(), translation=self.randomVector())
        inv = t.inverse()
        
        self.assertTrue(inv(t).is_identity())
    
    def test_roto_translation(self):
        t = Affine(rotation_axis='z', rotation_deg=90, translation=(2,0,0))
        self.assertVecsEqual( t((1,0,0)), (2,1,0) )
        
    def test_scaling(self):
        t = Affine(scaling=(1,2,3))
        self.assertVecsEqual( t((1,1,1)), (1,2,3) )
        
        t = Affine(scaling=10)
        self.assertVecsEqual( t((1,2,3)), (10,20,30) )
    
    def test_orthogonality(self):
        self.assertTrue( Affine.identity().is_orthogonal() )
        
        self.assertTrue( Affine(
            rotation_axis=self.randomVector(),
            rotation_deg=numpy.random.rand()*360).is_orthogonal() )
            
        self.assertFalse( Affine(scaling=2.0).is_orthogonal() )
    

if __name__ == '__main__':
    unittest.main()
