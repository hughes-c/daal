/* file: Step3LocalCollectionInputId.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP3LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input objects for the implicit ALS training algorithm
 * in the third step of the distributed processing mode
 */
public final class Step3LocalCollectionInputId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the local input object identifier using the provided value
     * @param value     Value corresponding to the local input object identifier
     */
    public Step3LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the local input object identifier
     * @return Value corresponding to the local input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int partialModelBlocksToNodeId = 1;

    /**
    * %Input partial models for the implicit ALS training algorithm in the third step
    * of the distributed processing mode
    */
    public static final Step3LocalCollectionInputId partialModelBlocksToNode =
            new Step3LocalCollectionInputId(partialModelBlocksToNodeId); /*!< Identifier of the input object */
}
/** @} */
