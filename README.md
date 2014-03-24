pyboostcvconverter
==================

This is a minimalist header necessary for using porting C++ functions/classes using OpenCV (3.0) Mat as arguments directly (w/o explicit conversions) to python.

Compatibility
-----------------
This code is written for OpenCV 3.X (Not earlier versions)! Currently, it presumes Python 2.7, but will be upgraded as soon as Python 3 support will be integrated into OpenCV 3.

Disclaimer
-----------------
Certain things in the code might be excessive/unneeded, so if you know something is not needed, please make a pull request with an update. Also, conversion errors aren't handled politically correct (i.e. just generates an empty matrix), please let me know if that bothers you or you'd like to fix that.
The code has been tested for memory leaks. If you still find any errors, let me know by positing an issue! 

Usage
----------------
Here is a usage sample based on Yati Sagade's sample from https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst:

```python

import numpy
import matmul # your module, also the name of your compiled dynamic library file w/o the extension

a = numpy.array([[1., 2., 3.]])
b = numpy.array([[1.],
                 [2.],
                 [3.]])
print(matmul.mul(a, b)) # should print [[14.]]
print(matmul.mul2(a, b)) # should also print [[14.]]
```
Here is the C++ code for the sample matmul.so module:

```c++

#include <boost/python.hpp>
#include "CVBoostConverter.hpp"

using namespace boost::python;

PyObject*
mul(PyObject *left, PyObject *right)
{
	cv::Mat leftMat, rightMat;
	leftMat = bcvt::fromNDArrayToMat(left);
    rightMat = bcvt::fromNDArrayToMat(right);
    auto c1 = leftMat.cols, r2 = rightMat.rows;
    // Check that the 2-D matrices can be legally multiplied.
    if (c1 != r2)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Incompatible sizes for matrix multiplication.");
        throw_error_already_set();
    }
    cv::Mat result = leftMat * rightMat;

    PyObject* ret = bcvt::fromMatToNDArray(result);

    return ret;
}

//This one uses Mat directly, but we won't need to worry about the conversions
Mat
mul2(Mat leftMat, Mat rightMat)
{
    auto c1 = leftMat.cols, r2 = rightMat.rows;
    if (c1 != r2)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Incompatible sizes for matrix multiplication.");
        throw_error_already_set();
    }
    cv::Mat result = leftMat * rightMat;

    return result;
}

static void init_ar()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(matmul)
{
	  using namespace XM;
	  init_ar();
	
	  //initialize converters
	  to_python_converter<cv::Mat,
		    bcvt::matToNDArrayBoostConverter>();
	  bcvt::matFromNDArrayBoostConverter();
    
    //expose module-level functions
    def("mul", mul);
    def("mul2",mul2);
}
