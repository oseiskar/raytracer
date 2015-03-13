
import jinja2

environment = None

def get_environment():
    global environment
    if environment is None:
        environment = jinja2.Environment(\
            loader=jinja2.PackageLoader('clray', 'cl_templates'), \
            line_statement_prefix='###')
    return environment

def render(name, *args, **kwargs):
    return get_environment().get_template(name).render(*args, **kwargs)

