/* file: TrainingResultId.java */
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
 * @ingroup ridge_regression_training
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__TRAININGRESULTID"></a>
 * @brief Available identifiers of the result of ridge regression model-based training
 */
public final class TrainingResultId {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the training result object identifier using the provided value
     * @param value     Value corresponding to the training result object identifier
     */
    public TrainingResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training result object identifier
     * @return Value corresponding to the training result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ModelId = 0;

    /** Ridge regression model */
    public static final TrainingResultId model = new TrainingResultId(ModelId);
}
/** @} */
