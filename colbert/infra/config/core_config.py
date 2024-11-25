import os
import torch
import ujson
import dataclasses

from typing import Any
from collections import defaultdict
from dataclasses import dataclass, fields
from colbert.utils.utils import timestamp, torch_load_dnn

from utility.utils.save_metadata import get_metadata_only


@dataclass
class DefaultVal:
    val: Any

    def __hash__(self):
        return hash(repr(self.val))

    def __eq__(self, other):
        self.val == other.val


@dataclass
class CoreConfig:
    def __post_init__(self):
        """
        Source: https://stackoverflow.com/a/58081120/1493011
        """

        self.assigned = {}

        for field in fields(self):
            field_val = getattr(self, field.name)

            if isinstance(field_val, DefaultVal) or field_val is None:
                setattr(self, field.name, field.default.val)

            if not isinstance(field_val, DefaultVal):
                self.assigned[field.name] = True

    def assign_defaults(self):
        """
        Assigns default values to the fields of the configuration object.

        This method iterates over all the fields of the configuration object and sets each field's value to its default.
        It also marks each field as assigned in the `assigned` dictionary.

        Attributes:
            self (object): The configuration object containing fields and their default values.
        """
        for field in fields(self):
            setattr(self, field.name, field.default.val)
            self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        ignored = set()

        for key, value in kw_args.items():
            self.set(key, value, ignore_unrecognized) or ignored.update({key})

        return ignored

        """
        # TODO: Take a config object, not kw_args.

        for key in config.assigned:
            value = getattr(config, key)
        """

    def set(self, key, value, ignore_unrecognized=False):
        """
        Sets the value of a configuration key.

        Parameters:
        key (str): The configuration key to set.
        value (any): The value to assign to the configuration key.
        ignore_unrecognized (bool): If True, unrecognized keys will be ignored. Defaults to False.

        Returns:
        bool: True if the key was successfully set, otherwise raises an exception.

        Raises:
        Exception: If the key is unrecognized and ignore_unrecognized is False.
        """
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True

        if not ignore_unrecognized:
            raise Exception(f"Unrecognized key `{key}` for {type(self)}")

    def help(self):
        print(ujson.dumps(self.export(), indent=4))

    def __export_value(self, v):
        v = v.provenance() if hasattr(v, "provenance") else v

        if isinstance(v, list) and len(v) > 100:
            v = (f"list with {len(v)} elements starting with...", v[:3])

        if isinstance(v, dict) and len(v) > 100:
            v = (f"dict with {len(v)} keys starting with...", list(v.keys())[:3])

        return v

    def export(self):
        d = dataclasses.asdict(self)

        for k, v in d.items():
            d[k] = self.__export_value(v)

        return d
