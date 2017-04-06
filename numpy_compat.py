"""This module allows the user to wrap numpy (and eventually sparse)
arrays and pass them to and from c functions."""

import ctypes
import numpy

class wrapped_lib(object):
	"""Takes a string and wraps the named library, such that numpy
	arrays can be passed.

	The mapping is:
	
	array -> array.data, array.shape[0], array.shape[1], ...
	"""
	
	def __init__(self, library_name):
		"""Example: lib = wrapped_lib('lib.so')"""
		self.__lib = ctypes.cdll.LoadLibrary(library_name)

	def __getattr__(self, name):
		"""Wrap the c function and return it to the user."""
		func = getattr(self.__lib, name)
		return lambda *args: func(*wrapped_lib.__convert_args(args))
		
	@classmethod
	def __convert_args(cls, arg):
		"""Converts a list of arguments such that ndarrays are expanded."""
		if type(arg) == tuple:
			return reduce(tuple.__add__, (cls.__convert_args(xx) for xx in arg))
		elif type(arg) == numpy.ndarray: # TODO use instanceof
			# mapping: array -> data, shape[0], shape[1], etc.
			arg_type = numpy.ctypeslib.ndpointer(dtype=arg.dtype, flags='CONTIGUOUS')
			return (arg_type.from_param(arg),) + arg.shape
		else:
			return (arg,)

def main():
	lib = wrapped_lib('test.so')
	print lib
	print lib.test(89)
	print lib.test(1234)

	import util
	@util.timed_function
	def add_arrays_1(a, b):
		return a ** 2 + b ** 2

	@util.timed_function
	def add_arrays_2(a,b):
		c = numpy.empty_like(a)
		lib.add_arrays(a, b, c)
		return c
		
	nn = 100000000
	a = numpy.arange(nn, dtype=numpy.float64)
	b = numpy.arange(nn, dtype=numpy.float64)
	c1 = add_arrays_1(a, b)
	c2 = add_arrays_2(a, b)
	print numpy.allclose(c1, c2)

if __name__ == '__main__':
	main()