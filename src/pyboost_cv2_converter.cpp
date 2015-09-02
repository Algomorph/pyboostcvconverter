/*
 * CV2BoostConverter.cpp
 *
 *  Created on: May 21, 2015
 *      Author: Gregory Kramida
 *   Copyright: 2015 Gregory Kramida
 */
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#if CV_VERSION_EPOCH == 2 || (!defined CV_VERSION_EPOCH && CV_VERSION_MAJOR == 2)
namespace pbcvt{
using namespace cv;

//===================   ERROR HANDLING     =========================================================
static int failmsg(const char *fmt, ...)
		{
	char str[1000];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(str, sizeof(str), fmt, ap);
	va_end(ap);

	PyErr_SetString(PyExc_TypeError, str);
	return 0;
}

static PyObject* failmsgp(const char *fmt, ...)
		{
	char str[1000];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(str, sizeof(str), fmt, ap);
	va_end(ap);

	PyErr_SetString(PyExc_TypeError, str);
	return 0;
}

//===================   THREADING     ==============================================================

class PyAllowThreads
{
public:
	PyAllowThreads() :
			_state(PyEval_SaveThread()) {
	}
	~PyAllowThreads()
	{
		PyEval_RestoreThread(_state);
	}
private:
	PyThreadState* _state;
};

class PyEnsureGIL
{
public:
	PyEnsureGIL() :
			_state(PyGILState_Ensure()) {
	}
	~PyEnsureGIL()
	{
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

//===================   NUMPY ALLOCATOR FOR OPENCV     =============================================
class NumpyAllocator:
		public MatAllocator
{
public:
	NumpyAllocator() {
	}
	~NumpyAllocator() {
	}

	void allocate(int dims, const int* sizes, int type, int*& refcount,
			uchar*& datastart, uchar*& data, size_t* step) {
		PyEnsureGIL gil;

		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int) (sizeof(size_t) / 8);
		int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
						depth == CV_16U ? NPY_USHORT :
						depth == CV_16S ? NPY_SHORT :
						depth == CV_32S ? NPY_INT :
						depth == CV_32F ? NPY_FLOAT :
						depth == CV_64F ? NPY_DOUBLE : f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
		int i;
		npy_intp _sizes[CV_MAX_DIM + 1];
		for (i = 0; i < dims; i++) {
			_sizes[i] = sizes[i];
		}

		if (cn > 1) {
			_sizes[dims++] = cn;
		}

		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

		if (!o) {
			CV_Error_(CV_StsError,
					("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		}
        refcount = refcountFromPyObject(o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA((PyArrayObject*) o);
	}

	void deallocate(int* refcount, uchar*, uchar*) {
		PyEnsureGIL gil;
		if (!refcount)
			return;
		PyObject* o = pyObjectFromRefcount(refcount);
		Py_INCREF(o);
		Py_DECREF(o);
	}
};

//===================   ALLOCATOR INITIALIZTION   ==================================================
NumpyAllocator g_numpyAllocator;

//===================   STANDALONE CONVERTER FUNCTIONS     =========================================

PyObject* fromMatToNDArray(const Mat& m) {
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}

Mat fromNDArrayToMat(PyObject* o) {
	cv::Mat m;
	if (!PyArray_Check(o)) {
		failmsg("argument is not a numpy array");
		if (!m.data)
			m.allocator = &g_numpyAllocator;
	} else {
		PyArrayObject* oarr = (PyArrayObject*) o;

		bool needcopy = false, needcast = false;
		int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
		int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
					typenum == NPY_USHORT ? CV_16U :
					typenum == NPY_SHORT ? CV_16S :
					typenum == NPY_INT ? CV_32S :
					typenum == NPY_INT32 ? CV_32S :
					typenum == NPY_FLOAT ? CV_32F :
					typenum == NPY_DOUBLE ? CV_64F : -1;

		if (type < 0) {
			if (typenum == NPY_INT64 || typenum == NPY_UINT64
					|| type == NPY_LONG) {
				needcopy = needcast = true;
				new_typenum = NPY_INT;
				type = CV_32S;
			} else {
				failmsg("Argument data type is not supported");
				m.allocator = &g_numpyAllocator;
				return m;
			}
		}

#ifndef CV_MAX_DIM
		const int CV_MAX_DIM = 32;
#endif

		int ndims = PyArray_NDIM(oarr);
		if (ndims >= CV_MAX_DIM) {
			failmsg("Dimensionality of argument is too high");
			if (!m.data)
				m.allocator = &g_numpyAllocator;
			return m;
		}
		int size[CV_MAX_DIM + 1];
		size_t step[CV_MAX_DIM + 1];
		size_t elemsize = CV_ELEM_SIZE1(type);
		const npy_intp* _sizes = PyArray_DIMS(oarr);
		const npy_intp* _strides = PyArray_STRIDES(oarr);
		bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

		for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
			// these checks handle cases of
			//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
			//  b) transposed arrays, where _strides[] elements go in non-descending order
			//  c) flipped arrays, where some of _strides[] elements are negative
			if ((i == ndims - 1 && (size_t) _strides[i] != elemsize)
					|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
				needcopy = true;
		}

		if (ismultichannel && _strides[1] != (npy_intp) elemsize * _sizes[2])
			needcopy = true;
		if (needcopy) {

			if (needcast) {
				o = PyArray_Cast(oarr, new_typenum);
				oarr = (PyArrayObject*) o;
			} else {
				oarr = PyArray_GETCONTIGUOUS(oarr);
				o = (PyObject*) oarr;
			}

			_strides = PyArray_STRIDES(oarr);
		}

		for (int i = 0; i < ndims; i++) {
			size[i] = (int) _sizes[i];
			step[i] = (size_t) _strides[i];
		}

		// handle degenerate case
		if (ndims == 0) {
			size[ndims] = 1;
			step[ndims] = elemsize;
			ndims++;
		}
		if (ismultichannel) {
			ndims--;
			type |= CV_MAKETYPE(0, size[2]);
		}

		m = Mat(ndims, size, type, PyArray_DATA(oarr), step);

		if (m.data){
			m.refcount = refcountFromPyObject(o);
			if (!needcopy){
				m.addref(); // protect the original numpy array from deallocation
							// (since Mat destructor will decrement the reference counter)
			}
		};
		m.allocator = &g_numpyAllocator;
	}
	return m;
}

//===================   BOOST CONVERTERS     =======================================================

PyObject* matToNDArrayBoostConverter::convert(Mat const& m) {
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}

matFromNDArrayBoostConverter::matFromNDArrayBoostConverter() {
	boost::python::converter::registry::push_back(matFromNDArrayBoostConverter::convertible,
			matFromNDArrayBoostConverter::construct,
			boost::python::type_id<Mat>());
}

/// @brief Check if PyObject is an array and can be converted to OpenCV matrix.
void* matFromNDArrayBoostConverter::convertible(PyObject* object) {

	if (!PyArray_Check(object)) {
		return NULL;
	}
#ifndef CV_MAX_DIM
	const int CV_MAX_DIM = 32;
#endif
	PyArrayObject* oarr = (PyArrayObject*) object;

	int typenum = PyArray_TYPE(oarr);
	if (typenum != NPY_INT64 && typenum != NPY_UINT64 && typenum != NPY_LONG
			&& typenum != NPY_UBYTE && typenum != NPY_BYTE
			&& typenum != NPY_USHORT && typenum != NPY_SHORT
			&& typenum != NPY_INT && typenum != NPY_INT32
			&& typenum != NPY_FLOAT && typenum != NPY_DOUBLE) {
		return NULL;
	}
	int ndims = PyArray_NDIM(oarr); //data type not supported

	if (ndims >= CV_MAX_DIM) {
		return NULL; //too many dimensions
	}

	return object;
}

/// @brief Construct a Mat from an NDArray object.
void matFromNDArrayBoostConverter::construct(PyObject* object,
		boost::python::converter::rvalue_from_python_stage1_data* data) {

	namespace python = boost::python;
	// Object is a borrowed reference, so create a handle indicting it is
	// borrowed for proper reference counting.
	python::handle<> handle(python::borrowed(object));

	// Obtain a handle to the memory block that the converter has allocated
	// for the C++ type.
	typedef python::converter::rvalue_from_python_storage<Mat> storage_type;
	void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

	// Allocate the C++ type into the converter's memory block, and assign
	// its handle to the converter's convertible variable.  The C++
	// container is populated by passing the begin and end iterators of
	// the python object to the container's constructor.
	PyArrayObject* oarr = (PyArrayObject*) object;

	bool needcopy = false, needcast = false;
	int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
	int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
				typenum == NPY_USHORT ? CV_16U :
				typenum == NPY_SHORT ? CV_16S :
				typenum == NPY_INT ? CV_32S :
				typenum == NPY_INT32 ? CV_32S :
				typenum == NPY_FLOAT ? CV_32F :
				typenum == NPY_DOUBLE ? CV_64F : -1;

	if (type < 0) {
		needcopy = needcast = true;
		new_typenum = NPY_INT;
		type = CV_32S;
	}

#ifndef CV_MAX_DIM
	const int CV_MAX_DIM = 32;
#endif
	int ndims = PyArray_NDIM(oarr);

	int size[CV_MAX_DIM + 1];
	size_t step[CV_MAX_DIM + 1];
	size_t elemsize = CV_ELEM_SIZE1(type);
	const npy_intp* _sizes = PyArray_DIMS(oarr);
	const npy_intp* _strides = PyArray_STRIDES(oarr);
	bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

	for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
		// these checks handle cases of
		//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
		//  b) transposed arrays, where _strides[] elements go in non-descending order
		//  c) flipped arrays, where some of _strides[] elements are negative
		if ((i == ndims - 1 && (size_t) _strides[i] != elemsize)
				|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
			needcopy = true;
	}

	if (ismultichannel && _strides[1] != (npy_intp) elemsize * _sizes[2])
		needcopy = true;

	if (needcopy) {

		if (needcast) {
			object = PyArray_Cast(oarr, new_typenum);
			oarr = (PyArrayObject*) object;
		} else {
			oarr = PyArray_GETCONTIGUOUS(oarr);
			object = (PyObject*) oarr;
		}

		_strides = PyArray_STRIDES(oarr);
	}

	for (int i = 0; i < ndims; i++) {
		size[i] = (int) _sizes[i];
		step[i] = (size_t) _strides[i];
	}

	// handle degenerate case
	if (ndims == 0) {
		size[ndims] = 1;
		step[ndims] = elemsize;
		ndims++;
	}

	if (ismultichannel) {
		ndims--;
		type |= CV_MAKETYPE(0, size[2]);
	}
	if (!needcopy) {
		Py_INCREF(object);
	}

	cv::Mat* m = new (storage) cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
	if (m->data){
		m->refcount = refcountFromPyObject(object);
		if (!needcopy){
			m->addref(); // protect the original numpy array from deallocation
						 // (since Mat destructor will decrement the reference counter)
		}
	};

	m->allocator = &g_numpyAllocator;
	data->convertible = storage;
}

} //end namespace pbcvt

#endif

