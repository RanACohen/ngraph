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

#include <locale>
#include <fstream>
#include <memory>
#include <string>

#include "ie_net_builder.hpp"
#include "ie_net_functions.hpp"

#include <cnn_network_impl.hpp>

using namespace IENetAPI;
using namespace std;
using namespace InferenceEngine;


class InternalNetworkImpl : public InferenceEngine::details::CNNNetworkImpl {
public:
    InternalNetworkImpl() {
        precision = Precision::FP32;
    }

    void remove(const string &layer_name) {
        _layers.erase(layer_name);
    }

    bool hasLayer(const string &lname) const {
        return _layers.find(lname) != _layers.end();
    }

    void addData(const DataPtr &data) {
        _data[data->name] = data;
    }

    void addOutput(const DataPtr &data) {
        addData(data);
        _outputData[data->name] = data;
    }
};

IENetBuilder::IENetBuilder(const std::string &cs):_name(cs)
{
    network = new InternalNetworkImpl();
}

IENetBuilder::~IENetBuilder() {
    delete network;
    network = nullptr;
}

void IENetBuilder::add(const IELayer &ir_layer) {
    network->addLayer(ir_layer);
    _layers.push_back(ir_layer);
}

bool IENetBuilder::shouldRemove(const IELayer &l) {
    if (l->type != "Reshape") return false;
    return l->insData[0].lock()->getDims() == output(l)->getDims();
}

void IENetBuilder::process(const IELayer &layer) {
    if (network->hasLayer(layer->name)) return;
    add(layer);
    for (auto o : layer->outData) {
        network->addData(o);
        for (auto l : o->inputTo)
            process(l.second);
    }
}

void IENetBuilder::optimize() {
    for (auto it = _layers.begin(); it != _layers.end();) {
        auto l = *it;
        if (shouldRemove(l)) {
            // l-in -> (l,l-out) -> (b-in list) ===> a-out -> (b-in list)
            auto lin = l->input();
            auto lout = output(l);

            lin->inputTo.erase(l->name);

            auto lout_targets = lout->inputTo;
            for (auto i : lout_targets) {
                lin->inputTo[i.first] = i.second;
                // reaplce target input data from lout to lin
                for (auto &tar_inp : i.second->insData) {
                    if (tar_inp.lock() == lout) {
                        tar_inp = lin;
                        break;
                    }
                }
            }

            it = _layers.erase(it);
            network->remove(l->name);
        } else {
            ++it;
        }
    }
}

void IENetBuilder::build() {
    if (_processed) return;
    InputsDataMap inputs;
    network->getInputsInfo(inputs);
    for (auto i : inputs) {
        for (auto l : i.second->getInputData()->inputTo)
            process(l.second);
    }

    for (auto l : network->allLayers()) {
        process(l.second);
    }
    optimize();
    _processed = true;
}

InferenceEngine::ICNNNetwork *IENetBuilder::getNetwork() {
    build();
    return network;
}

void IENetBuilder::crateDotFile(std::ostream &dot) const {
    dot << "digraph g {\n\tgraph[rankdir = \"LR\"];" << std::endl;

    for (auto &kvp : network->allLayers()) {
        saveLayerToDot(dot, kvp.second);
    }
    dot << std::endl << std::endl;
    for (auto &kvp : _edges) {
        dot << "\t\"layer_" << kvp.from.lid << "\":p" << kvp.from.pid << " -> \"layer_" << kvp.to.lid << "\":p"
            << kvp.to.pid << " [];" << std::endl;
    }
    dot << "}" << std::endl;
}



InferenceEngine::InputInfo::Ptr IENetBuilder::createInput(const std::string &name, const TensorDims &dims) const {
    Layout layout = NCHW;
    if (dims.size() == 2) {
        layout = InferenceEngine::Layout::NC;
    }
    TensorDesc td(InferenceEngine::Precision::FP32, dims, layout);
    auto inputData = std::make_shared<InferenceEngine::Data>(name, td);
    InferenceEngine::InputInfo::Ptr info(new InferenceEngine::InputInfo());
    info->setInputData(inputData);
    network->setInputInfo(info);
    return info;
}


void IENetBuilder::addOutput(const DataPtr &src) {
    network->addOutput(src);
}

void IENetBuilder::saveLayerToDot(std::ostream &dot, const IELayer &irLayer) const {
    /*
    "layer_4" [
    label = "name| type | <f2> |-1"
    shape = "record"
    ];
    */
    dot << "\t\"layer_" << irLayer->userValue.v_int << "\" [ label = \"" << _name << "| type: " << irLayer->type;
    int pid = 0;
    for (auto &in : irLayer->insData) {
        dot << "| ";
        auto dims = in.lock()->getDims();
        dot << "<p" << (pid++) << "> " << dims[0];
        for (int i = 1; i < dims.size(); ++i) dot << ", " << dims[i];
    }
    for (auto &p : irLayer->outData) {
        dot << "| ";
        auto dims = p->getDims();
        dot << "<p" << p->userObject.v_int << "> " << dims[0];
        for (int i = 1; i < dims.size(); ++i) dot << ", " << dims[i];
    }
    dot << "\"";
    dot << "\t\tshape = \"record\" ];" << std::endl;
}

void IENetBuilder::setName(const char *name) {
    _name = name;
}

