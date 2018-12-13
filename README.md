PBCVT (Python-Boost-OpenCV Converter)
==================

This is minimalist C++ sample code for porting C++ functions/classes using OpenCV Mat as arguments directly (w/o explicit conversions) to python. It was originally inspired by [code by Yati Sagade](https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst). 

It is mainly intended to be used as boilerplate code for your own C++ libraries which use OpenCV and which you'd like to call from Python code or a Python shell. Directly linking the generated library to other C++ code statically or dynamically is _not_ supported / has _not_ been tested. 

Compatibility
-----------------
This code is compatible with OpenCV 2.X, 3.X, and 4.X.
This code supports Python 2.7 and Python 3.X. You can pick one by setting PYTHON_DESIRED_VERSION to 3.X or 2.X in CMake.

Disclaimer
-----------------
Certain things in the code might be excessive/unneeded, so if you know something is not needed, please make a pull request with an update. Also, conversion errors aren't handled politically correct (i.e. just generates an empty matrix), please let me know if that bothers you or you'd like to fix that.
The code has been tested for memory leaks. If you still find any errors, let me know by positing an issue! 

Compiling & Trying Out Sample Code
----------------------
1. Install CMake and/or CMake-gui (http://www.cmake.org/download/, ```sudo apt-get install cmake cmake-gui``` on Ubuntu/Debian)
2. Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
3. Build (run the appropriate command ```make``` or ```ninja``` depending on your generator, or issue "Build All" on Windows+MSVC)
4. On *nix systems, ```make install``` run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version). On Windows+MSVC, build the INSTALL project.
5. Run python interpreter of your choice, issue the following commands:
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

Usage
----------------
The header and the two source files need to be directly included in your project. Use the provided CMake as an example to properly detect your & link python, numpy, and boost, as well as make a proper install target for your project. Use the python_module.cpp for an example of how to organize your own module. All repository sources may serve well as project boilerplate. Linking (statically or dynamically) to the actual example module is possible, but not recommended. **Windows users: please see note after the examples below.** **Troubleshooting CMake issues for older boost: also see note at the end.**

Here is (some of the) C++ code in the sample pbcvt.so module (python_module.cpp):

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
Notes for Windows Usage / Known Problems / Troubleshooting
----------------
When building on windows, please make sure to go over the following checklist.
- You have environment variable OpenCV_DIR set to the location of OpenCVModules.cmake file, e.g. ```C:\opencv\build\x64\vc15\lib``` in order for CMake to find OpenCV right away. "vc15" corresponds to the VisualStudio 2017, "vc14" to VS 2015, choose the one that matches your version.
- You have the directory containing opencv_world<your OpenCV version>.dll, e.g. C:\opencv\build\x64\vc15\bin, in your Path environment variable.
- You have boost properly built or downloaded as *static libraries* with *static runtime off*. A dynamic build would produce binaries such as boost_python37-vc141-mt-x64-1_68.lib and boost_python37-vc141-mt-x64-1_68.dll (**not what you need, notice the absense of the 'lib' prefix**), a static build with static runtime would produce files such as libboost_python37-vc141-mt-s-x64-1_68.lib (**not what you need, notice the 's' suffix**). What you need are files in the form: "libboost_python37-vc141-mt-x64-1_68.lib". If you're building boost from scratch, this command worked for me with Boost 1.68, 64-bit, Visual Studio 2017: ```b2 toolset=msvc-14.1 release debug runtime-link=shared link=static --build-type=complete --abbreviate-paths architecture=x86 address-model=64 install -j4```.
- Your boost directory is structured as follows: <your choice of Boost_DIR> which contains "lib" and "include" folders inside it. The include directory should have a "boost" subdirectory with all the headers, *not boost-1_68/boost* as is done by the build automatically. The path that you choose for Boost_DIR should *also* be in your environment variables.
- The memory address model / architecture (64 bit vs 32 bit, "x86_64" vs "x86") for all your binaries agree, i.e. your python installation needs to be a 64-bit one if your boost libraries have the "x64" suffix, likewise for your OpenCV, and finally for your choice of generator (i.e. Visual Studio ... Win64 for 64-bit) in CMake.

**Troubleshooting note on python37_d.lib**: I am still at war with Windows on having the Debug configuration done 100% correctly. You might still need it for such cases as, for instance, you have C++ unit tests which test your library and you want to debug through the unit test case. It *is* possible to do it right now, but there is an issue that sometimes requires a work-around. I got it to work by (1) installing the debug version of python through the official installer and (2) manually linking to the non-debug library in the debug project configuration within MSVC after the smake generation.

**Friendly reminder**: don't forget to build the INSTALL project in MSVC before trying to import your library in python.
    
Credits
----------------
Original code was inspired by [Yati Sagade's example](https://github.com/yati-sagade/blog-content/blob/master/content/numpy-boost-python-opencv.rst).
