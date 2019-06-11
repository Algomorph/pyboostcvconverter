//  ================================================================
//  Created by Gregory Kramida on 6/11/19.
//  Copyright (c) 2019 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace test_namespace {

namespace bp = boost::python;

#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
	static void init_ar(){
#endif
	Py_Initialize();

	import_array();
	return NUMPY_IMPORT_ARRAY_RETVAL;
}


/**
 * @brief Example function. Basic inner matrix product using explicit matrix conversion.
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
		bp::throw_error_already_set();
	}
	cv::Mat result = leftMat * rightMat;
	PyObject *ret = pbcvt::fromMatToNDArray(result);
	return ret;
}
/**
 * @brief Example function. Simply makes a new CV_16UC3 matrix and returns it as a numpy array.
 * @return The resulting numpy array.
 */

PyObject* makeCV_16UC3Matrix(){
	cv::Mat image = cv::Mat::zeros(240,320, CV_16UC3);
	PyObject* py_image = pbcvt::fromMatToNDArray(image);
	return py_image;
}

//
/**
 * @brief Example function. Basic inner matrix product using implicit matrix conversion.
 * @details This example uses Mat directly, but we won't need to worry about the conversion in the body of the function.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
	auto c1 = leftMat.cols, r2 = rightMat.rows;
	if (c1 != r2) {
		PyErr_SetString(PyExc_TypeError,
		                "Incompatible sizes for matrix multiplication.");
		bp::throw_error_already_set();
	}
	cv::Mat result = leftMat * rightMat;

	return result;
}

/**
 * \brief Example function. Increments all elements of the given matrix by one.
 * @details This example uses Mat directly, but we won't need to worry about the conversion anywhere at all,
 * it is handled automatically by boost.
 * \param matrix (numpy array) to increment
 * \return
 */
cv::Mat increment_elements_by_one(cv::Mat matrix){
	matrix += 1.0;
	return matrix;
}

//NOTE: the argument to the BOOST_PYTHON_MODULE has to correspond with the generated dynamic library file name.
//TO see how to avoid the "lib" prefix and append a standard python extension file suffix, see how the
//pbcvt library is handled in the CMakeLists.txt file in the root of the repository.
BOOST_PYTHON_MODULE (test_project) {
	//using namespace XM;
	init_ar();

	//initialize converters
	bp::to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
	pbcvt::matFromNDArrayBoostConverter();

	//expose module-level functions
	bp::def("dot", dot);
	bp::def("dot2", dot2);
	bp::def("makeCV_16UC3Matrix", makeCV_16UC3Matrix);

	//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
	//"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
	bp::def("increment_elements_by_one", increment_elements_by_one);
}

} // end namespace test_namespace