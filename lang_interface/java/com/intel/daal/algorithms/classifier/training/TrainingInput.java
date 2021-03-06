/* file: TrainingInput.java */
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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__TRAININGINPUT"></a>
 * @brief  %Input objects for the classifier training algorithm
 */
public class TrainingInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the input of the classifier training algorithm
     * @param context   Context to manage the input of the classifier training algorithm
     */
    public TrainingInput(DaalContext context) {
        super(context);
    }

    public TrainingInput(DaalContext context, long cInput) {
        super(context, cInput);
    }

    public TrainingInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm, cmode.getValue());
    }

    /**
     * Sets the input object for the classifier training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        if (id != InputId.data && id != InputId.labels) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object of the classifier training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id != InputId.data && id != InputId.labels) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int cmode);

    private native void cSetInput(long inputAddr, int id, long ntAddr);

    private native long cGetInput(long inputAddr, int id);
}
/** @} */
