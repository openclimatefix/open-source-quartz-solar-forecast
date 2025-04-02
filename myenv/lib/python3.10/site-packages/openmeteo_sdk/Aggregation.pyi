from __future__ import annotations

import flatbuffers
import numpy as np

import flatbuffers
import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class Aggregation(object):
  none: int
  minimum: int
  maximum: int
  mean: int
  p10: int
  p25: int
  median: int
  p75: int
  p90: int
  dominant: int
  sum: int
  spread: int

