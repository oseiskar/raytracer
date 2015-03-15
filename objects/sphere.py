from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm
import numpy
import math

class Sphere(Tracer):
    
    def __init__(self, pos, R):
        self.pos = tuple(pos)
        self.R = R
    
    def surface_area(self):
        return 4.0 * math.pi * self.R**2

    def random_surface_point_and_normal(self):
        p = numpy.array(self.pos)
        rand = numpy.random.normal(0,1,p.shape)
        rand = rand / numpy.linalg.norm(rand)
        return (p + rand * self.R, rand)
    
    def center_and_min_sampling_distance(self):
        return (numpy.array(self.pos), self.R * 2.0)
    
    @staticmethod
    def get_bounding_volume_code(center, R, minvar, maxvar):
        
        if R == None:
            return  """
            %s = 0.0f;
            %s = old_isec_dist;
            """ % (minvar, maxvar)
        
        # TOOD: trying to use this (or even calling without returning results
        # to some obscure bug, find out why...)
        #return Sphere(center, R)._make_code('bounding_volume(obj, "%s", "%s")' % (minvar, maxvar))
            
        code =  """
        {
        // Bounding sphere intersection
        
        const float R2 = %s;
        const float3 center = (float3)%s;
        float3 rel = center - origin;
        float dotp = dot(ray, rel);
        float psq = dot(rel, rel);
        """ % (R**2, tuple(center))
        
        code += """
        bool inside_bnd = psq < R2;
        
        if (dotp <= 0 && !inside_bnd) return;
        
        const float discr = dotp*dotp - psq + R2;
        if(discr < 0) return;
        const float sqrdiscr = native_sqrt(discr);
        
        %s = max(dotp-sqrdiscr,0.0f);
        %s = min(dotp+sqrdiscr,old_isec_dist);
        """ % (minvar,maxvar)
        
        code += """
        if (%s <= %s) return;
        }
        """  % (maxvar, minvar)
        
        return code

class SphereComponent(ConvexIntersection.Component):
    """Sphere"""
    
    n_subobjects = 1
    
    def __init__(self, pos, R):
        ConvexIntersection.Component.__init__(self,pos)
        self.R = R
    
