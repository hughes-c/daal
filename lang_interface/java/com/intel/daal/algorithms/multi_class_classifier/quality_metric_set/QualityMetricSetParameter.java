/* file: QualityMetricSetParameter.java */
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
 * @ingroup multi_class_classifier_quality_metric_set
 * @{
 */
/**
 * @brief Contains classes for computing the multi-class confusion matrix
 */
package com.intel.daal.algorithms.multi_class_classifier.quality_metric_set;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__QUALITYMETRICSETPARAMETER"></a>
 * @brief Class for the parameter of the multi-class SVM algorithm
 */

public class QualityMetricSetParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public QualityMetricSetParameter(DaalContext context, long cParameter, long nClasses) {
        super(context, cParameter);
        cSetNClasses(this.cObject, nClasses);
    }

    /**
     *  Gets the number of classes
     *  @return  Number of classes
     */
    public long getNClasses() {
        return cGetNClasses(this.cObject);
    }

    private native void cSetNClasses(long parAddr, long nClasses);

    private native long cGetNClasses(long parAddr);
}
/** @} */
