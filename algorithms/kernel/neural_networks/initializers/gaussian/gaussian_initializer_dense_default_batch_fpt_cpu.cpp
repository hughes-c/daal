/* file: gaussian_initializer_dense_default_batch_fpt_cpu.cpp */
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

//++
//  Implementation of gaussian calculation functions.
//--


#include "gaussian_initializer_batch_container.h"
#include "gaussian_initializer_kernel.h"
#include "gaussian_initializer_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{

namespace interface1
{
template class neural_networks::initializers::gaussian::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1

namespace internal
{
template class GaussianKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // internal

}
}
}
}
}
