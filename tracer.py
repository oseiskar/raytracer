
from utils import normalize_tuple, vec_norm
import numpy

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
        
    def _make_code(self, macro, template_env):
        s = "{% import '" + self.template_file_name() + "' as a %}" + \
            ("{{ a.%s }}\n" % macro)
        return template_env.from_string(s).render(obj=self)
    
    def make_tracer_function(self, template_env):
        return self._make_code('tracer_function(obj)', template_env)
        
    def make_normal_function(self, template_env):
        return self._make_code('normal_function(obj)', template_env)
    
    def make_functions(self, template_env):
        """
        Make necessary OpenCL functions for tracing objects of this class
        Returns a dictionary OpenCL function name -> function contents
        """
        
        return { \
            self.tracer_function_name : self.make_tracer_function(template_env),
            self.normal_function_name : self.make_normal_function(template_env)
        }


