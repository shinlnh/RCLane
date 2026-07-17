# Third-party notices

The native runtime links against NVIDIA TensorRT and the CUDA Runtime supplied
by the deployment environment. Those libraries are not distributed in this
repository and remain subject to NVIDIA's respective licenses.

Small compatibility routines were independently ported from these upstream
implementations so C++ reproduces the established Python pipeline:

- NumPy `npysort/aquicksort`, used to reproduce the tie ordering of
  `numpy.argsort`: NumPy is distributed under the BSD 3-Clause license.
  Source: <https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/npysort/quicksort.cpp>
- OpenCV fixed-point bilinear resize arithmetic, used to reproduce
  `cv2.dnn.blobFromImage`: OpenCV 4.11 is distributed under the Apache License
  2.0. Source:
  <https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/resize.cpp>

The full upstream license texts and copyright notices are available at the
linked projects.
