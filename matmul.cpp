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
      //using namespace XM;
      init_ar();

      //initialize converters
      to_python_converter<cv::Mat,
            bcvt::matToNDArrayBoostConverter>();
      bcvt::matFromNDArrayBoostConverter();

    //expose module-level functions
    def("mul", mul);
    def("mul2",mul2);
}
