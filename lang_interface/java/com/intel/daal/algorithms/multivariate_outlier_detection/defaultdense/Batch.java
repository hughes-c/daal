/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/**
 * @defgroup multivariate_outlier_detection_defaultdense Default method
 * @brief Contains classes for computing results of the multivariate outlier detection algorithm with the default method
 * @ingroup multivariate_outlier_detection
 * @{
 */
/**
 * @defgroup multivariate_outlier_detection_defaultdense_batch Batch
 * @ingroup multivariate_outlier_detection_defaultdense
 * @{
 */
/**
 * @brief Contains classes for computing the results of the multivariate outlier detection algorithm with the default method
 */
package com.intel.daal.algorithms.multivariate_outlier_detection.defaultdense;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.multivariate_outlier_detection.Input;
import com.intel.daal.algorithms.multivariate_outlier_detection.Method;
import com.intel.daal.algorithms.multivariate_outlier_detection.Result;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__DEFAULTDENSE__BATCH"></a>
 * \brief Runs multivariate outlier detection in the batch processing mode. \DAAL_DEPRECATED_USE{com.intel.daal.algorithms.multivariate_outlier_detection.Batch}
 * <!-- \n<a href="DAAL-REF-MULTIVARIATE_OUTLIER_DETECTION-ALGORITHM">multivariate outlier detection algorithm description and usage models</a> -->
 *
 * \par References
 *      - InputId class. Identifiers of the input objects for the multivariate outlier detection algorithm
 *      - ResultId class. Identifiers of the results of the multivariate outlier detection algorithm
 */
@Deprecated
public class Batch extends AnalysisBatch {
    public Input      input;     /*!< %Input data */
    public Parameter  parameter; /*!< Parameters of the algorithm */
    public Method     method;    /*!< Computation method for the algorithm */
    private Result    result;    /*!< %Result of the algorithm */
    private Precision prec;      /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs default multivariate outlier detection algorithm by copying input objects and parameters
     * of another algorithm default multivariate outlier detection algorithm
     * @param context   Context to manage the multivariate outlier detection
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs default multivariate outlier detection algorithm
     * @param context   Context to manage the multivariate outlier detection
     * @param cls       Data type to use in intermediate computations of the multivariate outlier detection,
     *                  Double.class or Float.class
     * @param method    Multivariate outlier detection computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs multivariate outlier detection algorithm
     * @return  Outlier detection results
     */
    @Override
    public Result compute() {
        super.compute();
        result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of the multivariate outlier detection results
     * @param result    Structure for storing the results of the multivariate outlier detection
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated default multivariate outlier detection algorithm
     * with a copy of input objects and parameters of this default multivariate outlier detection algorithm
     * @param context   Context to manage the multivariate outlier detection
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */