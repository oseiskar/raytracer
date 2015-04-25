
from accelerator import Accelerator
import numpy as np
import jinja2

class Compiler:
    
    def __init__(self, renderer):
        self.scene = renderer.scene
        self.shader = renderer.shader
        self.renderer = renderer
    
    def make_program(self):
        scene = self.scene
        
        template_env = jinja2.Environment(\
            loader=jinja2.PackageLoader('clray', 'cl_templates'),
            line_statement_prefix='###',
            trim_blocks=False,
            lstrip_blocks=False)
    
        kernels, functions = collect_tracer_kernels(scene.objects, template_env)
        function_declarations = [body[:body.find('{')] + ';' for body in functions]

        return template_env.get_template('main.cl').render({
            'shader': self.shader,
            'renderer': self.renderer,
            'objects': scene.objects,
            'n_objects': len(scene.objects),
            'functions': {
                'declarations': function_declarations,
                'definitions': functions,
                'kernels': kernels
            }
        })
    
def collect_tracer_kernels(objects, template_env):
    name_map = {}
    
    def push_func(name, body):
        if name in name_map and name_map[name].strip() != body.strip():
            print name_map[name]
            print '------'
            print body
            raise RuntimeError("function name clash!!")
    
    functions = set([])
    kernels = set([])
    
    for obj in objects:
        
        for (name, body) in obj.tracer.make_functions(template_env).items():
            push_func(name,body)
            functions.add(body)
            
        for (name, body) in obj.tracer.make_kernels(template_env).items():
            push_func(name,body)
            kernels.add(body)
    
    return (list(kernels),list(functions))
    
