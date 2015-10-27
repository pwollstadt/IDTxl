
# compile library source code into position-independent code (PIC)
gcc -c -Wall -Werror -fpic funs.c

# create a shared library from object file
gcc -shared -o libfuns.so funs.o

# run path, is a way of embedding the location of shared libraries in the executable itself, instead of relying on default locations or environment variables. We do this during the linking stage
#gcc -L/home/pwollsta/Dokumente/python_test/test_ctypes -Wl,-rpath=/home/pwollsta/Dokumente/python_test/test_ctypes -Wall -o test test_c_lib.c -lfuns -lm

# compile main program and link with shared library
# the current path may be written as ., also the linker needs -lm, i.e. an explicit flag for linking with math.h
gcc -L. -Wl,-rpath=. -Wall -o test_c test_c_lib.c -lfuns -lm

# list library dependencies
echo "dependencies of c main (using 'ldd'):"
ldd test_c		

# quick test using C code
./test_c

# running everything in Python
python import_c_funs.py

