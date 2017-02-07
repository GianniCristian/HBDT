# -*- coding: utf-8 -*-

import sys

class Bar(object):

    def __init__(self, max, status=""):

        self.status = status
        self.max = max
        self.dimension = 100
        self._current = 0
        self(self._current)

    def __call__(self, value, status=None):

        value = float(value)
        _current = int(value / self.max * self.dimension)
        if status:
            self.status = status
        if _current > self._current:
            wildcards = spaces = ""
            for _ in range(_current): wildcards = wildcards + "*"
            for _ in range(self.dimension-_current): wildcards = wildcards + " "
            bar = "\rSTATUS: " + self.status + " [" + wildcards + spaces + "]"
            sys.stdout.write(bar)
            sys.stdout.flush()
            self._current = _current
