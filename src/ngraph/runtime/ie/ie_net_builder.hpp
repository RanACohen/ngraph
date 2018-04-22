/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include "ie_net_layer.hpp"
#include "ie_icnn_network.hpp"

#include <vector>
#include <map>
#include <string>


class InternalNetworkImpl;

namespace IENetAPI {

class IENetBuilder {
private:
    struct Edge {
        struct port {
           int lid, pid;
        } from, to;
    };

    InternalNetworkImpl *network;
    std::vector<IELayer> _layers;  // ordered by input propagation
    std::vector<Edge> _edges;
    std::string _name;
    size_t _layer_id_cnt = 1;
    bool _processed = false;

    std::map<const void *, size_t> _segmentsMap;

    static bool shouldRemove(const IELayer &l);
    void process(const IELayer &value);
    void optimize();
    void build();

    void saveLayerToDot(std::ostream &dot, const IELayer &irLayer) const;
    IENetBuilder(IENetBuilder &) = delete;
    IENetBuilder operator=(IENetBuilder &) = delete;

public:
    explicit IENetBuilder(const std::string &cs);
    ~IENetBuilder();

    void add(const IELayer &ir_layer);

    // Products
    void crateDotFile(std::ostream &dot) const;

    DelayObj createDelay(const std::string &id, const TensorDims &dims);

    InferenceEngine::InputInfo::Ptr createInput(const std::string &name, const TensorDims &dims) const;

    void addOutput(const IELayer &src, int outIndx = 0);
    void addOutput(const InferenceEngine::DataPtr &src);
    void setName(const char *name);
    InferenceEngine::ICNNNetwork *getNetwork();
};

}  // namespace IRBuilder
