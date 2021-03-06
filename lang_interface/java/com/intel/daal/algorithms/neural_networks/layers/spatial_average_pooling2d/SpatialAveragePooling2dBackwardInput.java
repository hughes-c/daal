/* file: SpatialAveragePooling2dBackwardInput.java */
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
 * @defgroup spatial_average_pooling2d_backward Backward Two-dimensional Spatial pyramid average Pooling Layer
 * @brief Contains classes for backward spatial pyramid average 2D pooling layer
 * @ingroup spatial_average_pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__SPATIALAVERAGEPOOLING2DBACKWARDINPUT"></a>
 * @brief Input object for the backward two-dimensional spatial average pooling layer
 */
public final class SpatialAveragePooling2dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dBackwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public SpatialAveragePooling2dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward two-dimensional spatial average pooling layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SpatialAveragePooling2dLayerDataId id, NumericTable val) {
        if (id == SpatialAveragePooling2dLayerDataId.auxInputDimensions) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SpatialAveragePooling2dBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward two-dimensional spatial average pooling layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(SpatialAveragePooling2dLayerDataId id) {
        if (id == SpatialAveragePooling2dLayerDataId.auxInputDimensions) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
