import yaml


class Scenario:
    def __init__(self, path, runner):
        self.runner = runner
        self.events = list()
        with open(path, 'r') as file:
            events = yaml.load(file, Loader=yaml.Loader)
            for event in events:
                condition = event['condition']
                check_step = event['check_every']
                action = event['action']
                params = event['params']
                self.events.append(
                    {'condition': condition, 'check_step': check_step, 'action': action, 'params': params,
                     'done': False, 'last_check': None})

    def check_conditions(self):
        for event in self.events:
            step = self.get_attr(event['check_step'])
            if (event['last_check'] != step) and not event['done']:
                event['last_check'] = step
                attr_name, operator, val, repeat = event['condition']
                attr = self.get_attr(attr_name)
                if operator == 'equal':
                    if attr == val:
                        self.execute(event)
                elif operator == 'mod':
                    if (attr % val) == 0:
                        self.execute(event)
                else:
                    raise NotImplemented(f'Operator "{operator}" is not implemented!')

    def execute(self, event):
        method_name = event['action']
        params = event['params']
        f = self.get_attr(method_name)
        f(**params)
        if event['condition'][-1] == 'norepeat':
            event['done'] = True

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj
