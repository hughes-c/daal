/* file: logistic_layer.h */
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
//  Implementation of the logistic layer
//--
*/

#ifndef __LOGISTIC_LAYER_H__
#define __LOGISTIC_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/logistic/logistic_layer_types.h"
#include "algorithms/neural_networks/layers/logistic/logistic_layer_forward.h"
#include "algorithms/neural_networks/layers/logistic/logistic_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the logistic layer
 */
namespace logistic
{
namespace interface1
{
/**
 * @ingroup logistic_layers
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC__BATCH"></a>
 * \brief Provides methods for the logistic layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTICFORWARD-ALGORITHM">Forward logistic layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOGISTICBACKWARD-ALGORITHM">Backward logistic layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the logistic layer, double or float
 * \tparam method           Logistic layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public LayerIface
{
public:
    Parameter parameter; /*!< Logistic layer parameters */
    /** \brief Default constructor */
    Batch()
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace logistic
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
