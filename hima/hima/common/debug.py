from types import MethodType
from typing import Any


def inject_debug_tools(obj, inject_or_not=True):
    if isinstance(obj, type):
        return inject_debug_tools_to_type(obj, inject_or_not)
    else:
        inject_debug_tools_to_obj(obj, inject_or_not)


def inject_debug_tools_to_obj(obj, inject_or_not=True):
    """Create subclass with added debugging helpers via dynamic inheritance."""

    def set_breakpoint(self, method_name: str, callback):
        self.unset_breakpoint(method_name, callback)
        if method_name not in self._overrides:
            self._overrides[method_name] = [getattr(self, method_name)]

        self._overrides[method_name].append(callback)
        self._apply_breakpoint(method_name)

    def unset_breakpoint(self, method_name: str, callback):
        if method_name not in self._overrides or callback not in self._overrides[method_name]:
            return
        self._overrides[method_name].remove(callback)
        self._apply_breakpoint(method_name)

    def _apply_breakpoint(self, method_name: str):
        def wrap(breakpoint_callback, breakpointed_func):
            def wrapper(*args, **kwargs):
                return breakpoint_callback(self, breakpointed_func, *args, **kwargs)
            return wrapper

        overrides = self._overrides[method_name]
        method = overrides[0]
        for callback in overrides[1:]:
            method = wrap(callback, method)

        setattr(self, method_name, method)

    if not inject_or_not or hasattr(obj, 'set_breakpoint'):
        return

    obj._overrides = dict()
    setattr(obj, 'set_breakpoint', MethodType(set_breakpoint, obj))
    setattr(obj, 'unset_breakpoint', MethodType(unset_breakpoint, obj))
    setattr(obj, '_apply_breakpoint', MethodType(_apply_breakpoint, obj))


def remove_debug_tools_from_obj(obj):
    if not hasattr(obj, 'set_breakpoint'):
        return

    # noinspection PyProtectedMember
    overrides = obj._overrides
    for method_name in overrides:
        setattr(obj, method_name, overrides[method_name][0])

    delattr(obj, '_overrides')
    delattr(obj, 'set_breakpoint')
    delattr(obj, 'unset_breakpoint')
    delattr(obj, '_apply_breakpoint')


def inject_debug_tools_to_type(base_type, inject_or_not=True):
    """Create subclass with added debugging helpers via dynamic inheritance."""

    class DynamicWrapper(base_type):
        _overrides: dict[str, list[Any]]

        def __init__(self, *args, **kwargs):
            super(DynamicWrapper, self).__init__(*args, **kwargs)
            self._overrides = dict()

        def set_breakpoint(self, method_name: str, callback):
            self.unset_breakpoint(method_name, callback)
            if method_name not in self._overrides:
                self._overrides[method_name] = [getattr(self, method_name)]

            self._overrides[method_name].append(callback)
            self._apply_breakpoint(method_name)

        def unset_breakpoint(self, method_name: str, callback):
            if method_name not in self._overrides or callback not in self._overrides[method_name]:
                return
            self._overrides[method_name].remove(callback)
            self._apply_breakpoint(method_name)

        def _apply_breakpoint(self, method_name: str):
            def wrap(breakpoint_callback, breakpointed_func):
                def wrapper(*args, **kwargs):
                    return breakpoint_callback(self, breakpointed_func, *args, **kwargs)
                return wrapper

            overrides = self._overrides[method_name]
            method = overrides[0]
            for callback in overrides[1:]:
                method = wrap(callback, method)

            setattr(self, method_name, method)

    if not inject_or_not or issubclass(base_type, DynamicWrapper):
        return base_type

    return DynamicWrapper
