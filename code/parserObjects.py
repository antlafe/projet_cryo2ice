import argparse

from collections import OrderedDict


class ParentAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, default=OrderedDict(), **kwargs)

        self.children = []

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest)
        nspace = type(namespace)()
        for child in self.children:
            if child.default is not None:
                setattr(nspace, child.name, child.default)
        items[values] = nspace

class ChildAction(argparse.Action):
    def __init__(self, *args, parent, sub_action='store', **kwargs):
        super().__init__(*args, **kwargs)

        self.dest, self.name = parent.dest, self.dest
        self.action = sub_action
        self._action = None
        self.parent = parent

        parent.children.append(self)

    def get_action(self, parser):
        if self._action is None:
            action_cls = parser._registry_get('action', self.action, self.action)
            self._action = action_cls(self.option_strings, self.name)
        return self._action

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest)
        try:
            last_item = next(reversed(items.values()))
        except StopIteration:
            raise argparse.ArgumentError(self, "can't be used before {}".format(self.parent.option_strings[0]))
        action = self.get_action(parser)
        action(parser, last_item, values, option_string)
