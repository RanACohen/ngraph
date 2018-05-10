//
// Created by ranc on 4/24/18.
//
#include "ng_ie_bridge.hpp"

void convertAdd(nGraphIEBridge *bridge, const shared_ptr<Node> &op)
{
    const deque<descriptor::Input> &inputs = op->get_inputs();
    if (op->get_output_size()!=1) THROW("Add must have one output");
    if (op->get_input_size()==0) THROW("Add must have at least one input");
    if (op->get_input_size()==1) {
        bridge->addPort(op->get_instance_id(),0, bridge->getPortFromInput(inputs[0])); // skip this node
        return;
    }
    auto layer = IENetAPI::SumLayer::create(bridge->getPortFromInput(inputs[0]), bridge->getPortFromInput(inputs[1]));
    for (size_t i=2; i<op->get_input_size(); i++)
    {
        IENetAPI::addInput(layer, bridge->getPortFromInput(inputs[i]));
    }
    bridge->addPort(op->get_instance_id(),0,IENetAPI::output(layer));
}


std::map<std::string, std::function<void(nGraphIEBridge *bridge, const shared_ptr<Node> &op)>> nGraphIEBridge::sm_translators = {
        {"Add", convertAdd}
};
