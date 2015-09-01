import numpy
from transformations import Affine, rotation_matrix

class Tracer(object):
    """
    A Tracer instance represents the shape of a three-dimensional body.
    
    It is also associated with OpenCL code (templates) that can be used
    to compute the intersection of a ray and this object and an exterior
    normal at that intersection.
    """
    
    def __init__(self, position=(0,0,0), **affine_kwargs):
        # setting this to, e.g., id(obj), causes a tracer function to be
        # generated for each instance, instead of one per Tracer (sub)class
        self.unique_tracer_id = ""
        self.coordinates = Affine(translation=position, **affine_kwargs)
    
    def _function_name_prefix(self):
        return self.__class__.__name__+self.unique_tracer_id
    
    @property
    def tracer_function_name(self):
        return self._function_name_prefix() + '_tracer'
    
    @property
    def normal_function_name(self):
        return self._function_name_prefix() + '_normal'
        
    @property
    def tracer_kernel_name(self):
        return self._function_name_prefix() + '_tracer_kernel'
    
    @property
    def shadow_kernel_name(self):
        return self._function_name_prefix() + '_shadow_kernel'
        
    @property
    def normal_kernel_name(self):
        return self._function_name_prefix() + '_normal_kernel'
    
    def template_file_name(self):
        return 'objects/%s.cl' % self.template_name()
    
    def template_name(self):
        return self.__class__.__name__
        
    def _make_code(self, macro, template_env):
        s = "{% import '" + self.template_file_name() + "' as a %}" + \
            ("{{ a.%s }}\n" % macro)
        return template_env.from_string(s).render(obj=self)
    
    def make_functions(self, template_env):
        """
        Make necessary OpenCL functions for tracing objects of this class
        Returns a dictionary OpenCL function name -> function contents
        """
        
        return { \
            self.tracer_function_name : \
                self._make_code('tracer_function(obj)', template_env),
            self.normal_function_name : \
                self._make_code('normal_function(obj)', template_env)
        }

    def make_kernels(self, template_env):
        """
        Make necessary OpenCL kernels for tracing objects of this class
        Returns a dictionary OpenCL kernel name -> function contents
        """
        
        return { \
            self.tracer_kernel_name : \
                self._make_code('tracer_kernel(obj)', template_env),
            self.shadow_kernel_name : \
                self._make_code('shadow_kernel(obj)', template_env),
            self.normal_kernel_name : \
                self._make_code('normal_kernel(obj)', template_env)
        }
        
    def parameter_declarations(self):
        return []
    
    def parameter_values(self):
        return [getattr(self, name) for _, name in self._typed_parameters()]
    
    def parameter_declaration_string(self):
        params = self.parameter_declarations()
        if self.has_data():
            params = [
                    "TRACER_DATA const float4 *vector_data",
                    "TRACER_DATA const int *integer_data"
                ] + params
        return cl_parameter_string(params)
    
    def parameter_types(self):
        return [cl_type for cl_type, _ in self._typed_parameters()]
    
    def parameter_names(self):
        return [name for _, name in self._typed_parameters()]
    
    def _typed_parameters(self):
        for p in self.parameter_declarations():
            cl_type, name = p.split()
            yield((cl_type,name))

    def has_data(self):
        return hasattr(self, 'get_data')
    
    def tracer_coordinate_system(self):
        return Affine.identity()
    
    def global_to_tracer_coordinate_transform(self):
        return self.coordinates(self.tracer_coordinate_system())
    
    @property
    def auto_flip_normal(self):
        """
        Automatically flip computed normal to correct direction?
        """
        return False
    
    @property
    def convex(self):
        return False
    
    def linear_transform(self, linear=None, translation=None, **kwargs):
        if translation is not None:
            raise RuntimeError('translation not allowed')
        
        transform = Affine(linear=linear, **kwargs).linear
        
        self.coordinates = Affine(
            linear=numpy.dot(transform, self.coordinates.linear),
            translation=self.coordinates.translation)
        return self
        
    def rotate(self, axis, deg):
        return self.linear_transform(rotation_axis=axis, rotation_deg=deg)
    
    @property
    def position(self):
        return self.coordinates.translation
    
    @position.setter
    def position(self, value):
        self.coordinates = Affine(
            linear=self.coordinates.linear,
            translation=value
        )

def cl_parameter_string(params):
    if len(params) == 0: return ''
    else: return ', '.join([''] + params)
