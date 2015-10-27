import ctypes
bib = ctypes.CDLL("/home/pwollsta/Dokumente/python_test/test_ctypes/libfuns.so") 

print "Testing functions in Python:\n"

# test fakultaet
print "######################## fakultaet()"
print "The faculty of 4:"
print bib.fakultaet(4)

# test vector length
v1 = ctypes.c_double(3.5)
v2 = ctypes.c_double(7.4)
v3 = ctypes.c_double(1.2)
bib.veclen.restype = ctypes.c_double
vec_len = bib.veclen(v1,v2,v3)
print "######################## veclen()"
print "vector length: %.2f" % vec_len

# test bubblesort
arraytyp = ctypes.c_int * 10 
a = arraytyp(0,2,5,2,8,1,4,7,3,8) 
print "######################## sortiere()"
print "Vorher: ", [i for i in a] 
bib.sortiere(a, 10)
print "Nachher: ", [i for i in a]
