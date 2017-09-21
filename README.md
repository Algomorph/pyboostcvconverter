Integration
==================

1.  Add pyboostcvconverter to package.xml
2.  Add ```pyboostcvconverter/cmake/DetectPython.cmake``` to your module's ```cmake/Modules/``` folder
3.  Add find package to CMakeLists.txt
  1. Add find package

```
find_package(catkin REQUIRED COMPONENTS
  ...  
  pyboostcvconverter
  ...
)
```
  2.  Add python setup to CMakeLists.txt
```
####################
#====== PYTHON Interface
####################

set(PYTHON_OPTIONS "2.X" "3.X")
set(PYTHON_DESIRED_VERSION "2.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})

#=============== Find Packages 
find_package(OpenCV COMPONENTS core REQUIRED)
include("DetectPython")

set(Python_ADDITIONAL_VERSIONS ${PYTHON2_VERSION_MAJOR}.${PYTHON2_VERSION_MINOR})
find_package(Boost COMPONENTS python-py${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR} REQUIRED)

#========pick python stuff========================================

SET(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
SET(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
SET(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
SET(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
SET(ARCHIVE_OUTPUT_NAME pbcvt_py2)
```
3. create your library with name ```X```, link and add dependencies to CMakeLists.txt
4. Make sure in your c++ boost code the name of your python module is ```libX``` and that you 

```c++

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

using namespace boost::python;

... your code ...

static void init_ar(){
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

BOOST_PYTHON_MODULE (libmylibname) {
    //using namespace XM;
    init_ar();

    //initialize converters
    to_python_converter<cv::Mat,
            pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    //expose module-level functions
    def("myfn", myfn);

}

```

pyboostcvconverter
==================

This is minimalist C++ code for porting C++ functions/classes using OpenCV Mat as arguments directly (w/o explicit conversions) to python. Originally inspired by [code by Yati Sagade](https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst).

Compatibility
-----------------
(Update) This code is now compatible OpenCV 2.X and 3.X. 
(Update) This code now supports Python 2.7 and ~~experimentally~~ Python 3.X. You can pick one by setting PYTHON_DESIRED_VERSION to 3.X or 2.X in CMake.

Disclaimer
-----------------
Certain things in the code might be excessive/unneeded, so if you know something is not needed, please make a pull request with an update. Also, conversion errors aren't handled politically correct (i.e. just generates an empty matrix), please let me know if that bothers you or you'd like to fix that.
The code has been tested for memory leaks. If you still find any errors, let me know by positing an issue! 

Compiling & Trying Out Sample Code
----------------------
1. Install CMake and/or CMake-gui (http://www.cmake.org/download/, ```sudo apt-get install cmake cmake-gui``` on Ubuntu/Debian)
2. Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
3. Build (run ```make``` on *nix systems with gcc/eclipse CDT generator from within the build folder)
4. On *nix systems, ```make install``` run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version).
5. Run python interpreter of your choice, issue 
  1. import pbcvt; import numpy as np
  2. a = np.array([[1.,2.],[3.,4.]]); b = np.array([[2.,2.],[4.,4.]])
  3. pbcvt.dot(a,b)
  4. pbcvt.dot2(a,b)

Usage
----------------
The header and the two source files need to be directly included in your project. Use the provided CMake as an example to properly detect your & link python, numpy, and boost, as well as make a proper install target for your project. Use the python_module.cpp for an example of how to organize your own module. All repository sources may serve well as project boilerplate. Linking (statically or dynamically) to the actual example module is possible, but not recommended.

```python

import numpy
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extension

a = numpy.array([[1., 2., 3.]])
b = numpy.array([[1.],
                 [2.],
                 [3.]])
print(pbcvt.dot(a, b)) # should print [[14.]]
print(pbcvt.dot2(a, b)) # should also print [[14.]]
```
Here is the C++ code for the sample pbcvt.so module (python_module.cpp):

```c++
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace pbcvt {

    using namespace boost::python;

/**
 * Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(result);
        return ret;
    }

//This example uses Mat directly, but we won't need to worry about the conversion
/**
 * Example function. Basic inner matrix product using implicit matrix conversion.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,
                pbcvt::matToNDArrayBoostConverter>();
        pbcvt::matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);

    }

} //end namespace pbcvt
```

Original code is based on [yati sagade's sample](https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst) but has since been heavily revised and upgraded.
