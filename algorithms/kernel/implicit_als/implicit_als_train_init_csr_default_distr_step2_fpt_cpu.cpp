/* file: implicit_als_train_init_csr_default_distr_step2_fpt_cpu.cpp */
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

/*
//++
//  Implementation of implicit ALS initialization functions for default CSR method.
//--
*/

#include "implicit_als_train_init_kernel.h"
#include "implicit_als_train_init_csr_default_distr_impl.i"
#include "implicit_als_train_init_container.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace interface1
{
template class DistributedContainer<step2Local, DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class ImplicitALSInitDistrStep2Kernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
}
}
}
}
}
