//
// Created by ranc on 4/23/18.
//

#include "ng_ie_bridge.hpp"

void nGraphIEBridge::addLayer(const shared_ptr<Node> &op)
{
    std::string node_op = op->description();

    auto it = sm_translators.find("Add");
    if (it != sm_translators.end())
    {
        it->second(this, op);
        return;
    }

    // fallback - automatic conversion using 1:1 mapping
    op->get_arguments();
    
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
