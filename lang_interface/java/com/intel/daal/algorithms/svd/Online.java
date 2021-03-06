/* file: Online.java */
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
 * @defgroup svd_online Online
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.algorithms.AnalysisOnline;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINE"></a>
 * @brief Runs the SVD algorithm in the online processing mode
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of SVD input objects
 *      - PartialResultId class. Identifiers of partial SVD results
 *      - ResultId class. Identifiers of SVD results
 *      - ResultFormat class. Options to return SVD output matrices
 */
public class Online extends AnalysisOnline {
    public Input                  input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method     method;  /*!< Computation method for the algorithm */
    protected OnlinePartialResult partialResult;    /*!< Partial result of the algorithm */
    protected Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Online(DaalContext context, Online other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        partialResult = null;
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param cls       Data type to use in intermediate computations for the SVD algorithm,
     *                  Double.class or Float.class
     * @param method    SVD computation method, @ref Method
     */
    public Online(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (this.method != Method.defaultDense) {
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

        this.cObject = cInitOnline(prec.getValue(), method.getValue());
        partialResult = null;
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the SVD algorithm
     * @return  Partial results of the SVD algorithm obtained in the online processing mode
     */
    @Override
    public OnlinePartialResult compute() {
        super.compute();
        partialResult = new OnlinePartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes final results of the SVD algorithm
     * @return  Final results of the SVD algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of the SVD algorithm
     * @param partialResult         Structure to store partial results of the SVD algorithm
     * @param initializationFlag    Flag that specifies whether partial results are initialized
     */
    public void setPartialResult(OnlinePartialResult partialResult, boolean initializationFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(),
                initializationFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of the SVD algorithm
     * @param partialResult         Structure to store partial results of the SVD algorithm
     */
    public void setPartialResult(OnlinePartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory to store results of the SVD algorithm
     * @param result    Structure to store results of the SVD algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * @param context   Context to manage created SVD algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Online clone(DaalContext context) {
        return new Online(context, this);
    }

    private native long cInitOnline(int prec, int method);

    private native long cInitParameter(long addr, int prec, int method);

    private native long cGetInput(long addr, int prec, int method);

    private native long cGetResult(long addr, int prec, int method);

    protected native long cGetPartialResult(long addr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult,
            boolean initializationFlag);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
