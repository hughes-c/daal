/* file: SpatialMaximumPooling2dBackwardResult.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup spatial_maximum_pooling2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_maximum_pooling2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_MAXIMUM_POOLING2D__SPATIALMAXIMUMPOOLING2DBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward two-dimensional spatial maximum pooling layer
 */
public class SpatialMaximumPooling2dBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dBackwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward two-dimensional spatial maximum pooling layer
     * @param context   Context to manage the backward two-dimensional spatial maximum pooling layer
     */
    public SpatialMaximumPooling2dBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public SpatialMaximumPooling2dBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
