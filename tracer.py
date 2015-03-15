
from utils import normalize_tuple, vec_norm
import numpy
import templates

class Tracer:
    """
    A Tracer instance represents the shape of a three-dimensional body.
    It is responsible for generating the OpenCL code that can compute the
    intersection of a ray and this object (given helpful extra information
    that is accumulated during the tracing process) and an exterior normal
    at that intersection.
    """

    # setting this to, e.g., id(obj), causes a tracer function to be
    # generated for each instance, instead of one per Tracer (sub)class
    unique_tracer_id = ""
    
    def _function_name_prefix(self):
        return self.__class__.__name__+self.unique_tracer_id
    
    @property
    def tracer_function_name(self):
        return self._function_name_prefix() + '_tracer'
    
    @property
    def normal_function_name(self):
        return self._function_name_prefix() + '_normal'
    
    def template_file_name(self):
        return 'objects/%s.cl' % self.template_name()
    
    def template_name(self):
        return self.__class__.__name__
        
    def _make_code(self, macro):
        s = "{% import '" + self.template_file_name() + "' as a %}" + \
            ("{{ a.%s }}\n" % macro)
        return templates.get_environment().from_string(s).render(obj=self)
    
    def make_tracer_function(self):
        return self._make_code('tracer_function(obj)')
        
    def make_normal_function(self):
        return self._make_code('normal_function(obj)')
        
    def make_tracer_call(self, base_params):
        """
        Make a call that computes the intersection of given ray and an object
        represented by this tracer instance (returns a string of OpenCL code)
        """
        return self._make_code('tracer_call(obj, "%s")' % base_params)
        
    def make_normal_call(self, base_params):
        """
        Make a call that computes an exterior normal in the given intersection
        point (returns a string of OpenCL code)
        """
        return self._make_code('normal_call(obj, "%s")' % base_params)
    
    def make_functions(self):
        """
        Make necessary OpenCL functions for tracing objects of this class
        Returns a dictionary OpenCL function name -> function contents
        """
        
        return { \
            self.tracer_function_name : self.make_tracer_function(),
            self.normal_function_name : self.make_normal_function()
        }


