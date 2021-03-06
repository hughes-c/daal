/* file: TrainingResult.java */
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
 * @ingroup gbt_regression_training
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.gbt.regression.Model;
import com.intel.daal.algorithms.gbt.regression.training.TrainingResultId;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method
 *        of the gradient boosted trees regression training algorithm in the batch processing mode.
 */
public final class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public TrainingResult(DaalContext context, long cAlgorithm, Precision prec, ComputeMode cmode) {
        super(context);
        this.cObject = cGetResult(cAlgorithm, prec.getValue(), cmode.getValue());
    }

    /**
     * Returns the final result of the gradient boosted trees regression training algorithms
     * @param id   Identifier of the result
     * @return         %Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, TrainingResultId.model.getValue()));
    }
    private native long cGetResult(long algAddr, int prec, int method);
    private native long cGetModel(long resAddr, int id);
}
/** @} */
