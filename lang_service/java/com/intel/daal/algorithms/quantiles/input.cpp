/* file: input.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_quantiles_Offline */

#include "daal.h"
#include "quantiles/JInput.h"
#include "quantiles/JMethod.h"

#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::quantiles;

/*
 * Class:     com_intel_daal_algorithms_quantiles_Input
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_quantiles_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<quantiles::Input>::set<InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_quantiles_Input
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_quantiles_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<quantiles::Input>::get<InputId, NumericTable>(inputAddr, id);
}
