//
// Created by ranc on 4/23/18.
//

#include "ng_ie_bridge.hpp"

virtual void nGraphIEBridge::addLayer(const shared_ptr<Node> &op)
{
    std::string node_op = op->description();

    if (node_op == "Add")
    {
        const deque<descriptor::Input> &inputs = op->get_inputs();
        if (op->get_output_size()!=1) THROW("Add must have one output");
        if (op->get_input_size()==0) THROW("Add must have at least one input");
        if (op->get_input_size()==1) {
            addPort(op->get_instance_id(),0, getPortFromInput(inputs[0])); // skip this node
            return;
        }
        auto layer = IENetAPI::SumLayer::create(getPortFromInput(inputs[0]), getPortFromInput(inputs[1]));
        for (size_t i=2; i<op->get_input_size(); i++)
        {
            IENetAPI::addInput(layer, getPortFromInput(inputs[i]));
        }
        addPort(op->get_instance_id(),0,IENetAPI::output(layer));
    }
}

void nGraphIEBridge::convert(const Function::Ptr &function, IENetAPI::IENetBuilder &doc)
{
    size_t input_count = 0;
    for (auto param : function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            auto tv = param->get_output_tensor_view(i);
            const auto &shape = tv->get_tensor_view_type()->get_shape();
            addPort(param->get_instance_id(),i, doc.createInput(param->get_name(), shape)->getInputData());
            skip(param);
        }
    }

    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        if (shouldSkip(op)) continue;
        if (op->description() == "Parameter")
        {
            // we should have skipped it
            THROW("found a parameter in network but it is not in function parameters: ") << op->get_name();
        }
        // get the inputs of this op from portMap
        addLayer(op);
    }
}
