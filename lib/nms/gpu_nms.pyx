# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
cimport numpy as np

# Ensure size of int matches numpy int32
assert sizeof(int) == sizeof(np.int32_t)

# External C++ function declaration
cdef extern from "gpu_nms.hpp":
    void _nms(int*, int*, const float*, int, int, float, int)

def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out

    # Create a NumPy array for keep
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)

    # Get scores and order
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
    cdef np.ndarray[np.int32_t, ndim=1] order = scores.argsort()[::-1].astype(np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = dets[order, :]

    # Call the NMS function
    _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)

    # Return the results as a NumPy array
    return keep[:num_out]  # This is already a NumPy array
