/*
 * CV4BoostConverter.cpp
 *
 */
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#if CV_VERSION_MAJOR == 4
namespace pbcvt {
using namespace cv;
//===================   ERROR HANDLING     =========================================================

static int failmsg(const char *fmt, ...) {
	char str[1000];

	va_list ap;
	va_start(ap, fmt);
	vsnprintf(str, sizeof(str), fmt, ap);
	va_end(ap);

	PyErr_SetString(PyExc_TypeError, str);
	return 0;
}

//===================   THREADING     ==============================================================
class PyAllowThreads {
public:
	PyAllowThreads() :
			_state(PyEval_SaveThread()) {
	}
	~PyAllowThreads() {
		PyEval_RestoreThread(_state);
	}
private:
	PyThreadState* _state;
};

class PyEnsureGIL {
public:
	PyEnsureGIL() :
			_state(PyGILState_Ensure()) {
	}
	~PyEnsureGIL() {
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

enum {
	ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2
};

class NumpyAllocator:
		public MatAllocator {
public:
	NumpyAllocator() {
		stdAllocator = Mat::getStdAllocator();
	}
	~NumpyAllocator() {
	}

	UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
			size_t* step) const {
		UMatData* u = new UMatData(this);
		u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
		npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
		for (int i = 0; i < dims - 1; i++)
			step[i] = (size_t) _strides[i];
		step[dims - 1] = CV_ELEM_SIZE(type);
		u->size = sizes[0] * step[0];
		u->userdata = o;
		return u;
	}

	UMatData* allocate(int dims0, const int* sizes, int type, void* data,
			size_t* step, cv::AccessFlag flags, UMatUsageFlags usageFlags) const {
		if (data != 0) {
			CV_Error(Error::StsAssert, "The data should normally be NULL!");
			// probably this is safe to do in such extreme case
			return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
					usageFlags);
		}
		PyEnsureGIL gil;

		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int) (sizeof(size_t) / 8);
		int typenum =
				depth == CV_8U ? NPY_UBYTE :
				depth == CV_8S ? NPY_BYTE :
				depth == CV_16U ? NPY_USHORT :
				depth == CV_16S ? NPY_SHORT :
				depth == CV_32S ? NPY_INT :
				depth == CV_32F ? NPY_FLOAT :
				depth == CV_64F ?
									NPY_DOUBLE :
									f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
		int i, dims = dims0;
		cv::AutoBuffer<npy_intp> _sizes(dims + 1);
		for (i = 0; i < dims; i++)
			_sizes[i] = sizes[i];
		if (cn > 1)
			_sizes[dims++] = cn;
		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
		if (!o)
			CV_Error_(Error::StsError,
					("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		return allocate(o, dims0, sizes, type, step);
	}

	bool allocate(UMatData* u, cv::AccessFlag accessFlags,
			UMatUsageFlags usageFlags) const {
		return stdAllocator->allocate(u, accessFlags, usageFlags);
	}

	void deallocate(UMatData* u) const {
		if (u) {
			PyEnsureGIL gil;
			PyObject* o = (PyObject*) u->userdata;
			Py_XDECREF(o);
			delete u;
		}
	}

	const MatAllocator* stdAllocator;
};

//===================   ALLOCATOR INITIALIZTION   ==================================================
NumpyAllocator g_numpyAllocator;

//===================   STANDALONE CONVERTER FUNCTIONS     =========================================

PyObject* fromMatToNDArray(const Mat& m) {
	if (!m.data)
		Py_RETURN_NONE;
		Mat temp,
	*p = (Mat*) &m;
	if (!p->u || p->allocator != &g_numpyAllocator) {
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	PyObject* o = (PyObject*) p->u->userdata;
	Py_INCREF(o);
	return o;
}

Mat fromNDArrayToMat(PyObject* o) {
	cv::Mat m;
	bool allowND = true;
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

		if (ndims > 2 && !allowND) {
			failmsg("%s has more than 2 dimensions");
		} else {

			m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
			m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
			m.addref();

			if (!needcopy) {
				Py_INCREF(o);
			}
		}
		m.allocator = &g_numpyAllocator;
	}
	return m;
}

//===================   BOOST CONVERTERS     =======================================================

PyObject* matToNDArrayBoostConverter::convert(Mat const& m) {
	if (!m.data)
		Py_RETURN_NONE;
		Mat temp,
	*p = (Mat*) &m;
	if (!p->u || p->allocator != &g_numpyAllocator)
			{
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	PyObject* o = (PyObject*) p->u->userdata;
	Py_INCREF(o);
	return o;
}

matFromNDArrayBoostConverter::matFromNDArrayBoostConverter() {
	boost::python::converter::registry::push_back(convertible, construct,
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
	m->u = g_numpyAllocator.allocate(object, ndims, size, type, step);
	m->allocator = &g_numpyAllocator;
	m->addref();
	data->convertible = storage;
}

}			//end namespace pbcvt
#endif
