/* file: DistributedPartialResultStep2Id.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * @brief Available identifiers of partial results of the implicit ALS training algorithm obtained
 * in the second step of the distributed processing mode
 */
public final class DistributedPartialResultStep2Id {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep2Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep2ForStep4Id = 0;

/** Partial results of the implicit ALS training algorithm obtained in the second step of the distributed processing mode */
    public static final DistributedPartialResultStep2Id outputOfStep2ForStep4 = new DistributedPartialResultStep2Id(
            outputOfStep2ForStep4Id);
}
/** @} */
