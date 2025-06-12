from __future__ import annotations

import flatbuffers
import numpy as np

import flatbuffers
import typing

uoffset: typing.TypeAlias = flatbuffers.number_types.UOffsetTFlags.py_type

class Unit(object):
  undefined: int
  celsius: int
  centimetre: int
  cubic_metre_per_cubic_metre: int
  cubic_metre_per_second: int
  degree_direction: int
  dimensionless_integer: int
  dimensionless: int
  european_air_quality_index: int
  fahrenheit: int
  feet: int
  fraction: int
  gdd_celsius: int
  geopotential_metre: int
  grains_per_cubic_metre: int
  gram_per_kilogram: int
  hectopascal: int
  hours: int
  inch: int
  iso8601: int
  joule_per_kilogram: int
  kelvin: int
  kilopascal: int
  kilogram_per_square_metre: int
  kilometres_per_hour: int
  knots: int
  megajoule_per_square_metre: int
  metre_per_second_not_unit_converted: int
  metre_per_second: int
  metre: int
  micrograms_per_cubic_metre: int
  miles_per_hour: int
  millimetre: int
  pascal: int
  per_second: int
  percentage: int
  seconds: int
  unix_time: int
  us_air_quality_index: int
  watt_per_square_metre: int
  wmo_code: int
  parts_per_million: int

