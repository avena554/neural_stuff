from __future__ import print_function
import sys


def print_on_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)