//
// Created by ranc on 4/23/18.
//

#ifndef NGRAPH_NGRAPHIEBRIDGE_H
#define NGRAPH_NGRAPHIEBRIDGE_H


#include "ngraph/runtime/ie/ie_backend.hpp"

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"

#include "ie_net_builder.hpp"
#include "ie_net_functions.hpp"


using namespace std;
using namespace ngraph;

class nGraphIEBridge
{
    static std::map<std::string, std::function<void(nGraphIEBridge *bridge, const shared_ptr<Node> &op)>> sm_translators;
    std::map<std::pair<size_t,size_t>, IENetAPI::OutputPort> m_portMap;
    std::set<size_t> m_skipList;
public:
    void skip(const shared_ptr<op::Op> &op)
    {
        m_skipList.insert(op->get_instance_id());
    }

    bool shouldSkip(const shared_ptr<Node> &op) const
    {
        return m_skipList.find(op->get_instance_id()) != m_skipList.end();
    }

    bool isPortExist(size_t id, size_t indx) const
    {
        return m_portMap.find(std::make_pair(id, indx)) != m_portMap.end();
    }

    void addPort(size_t id, size_t indx, const IENetAPI::OutputPort &port)
    {
        m_portMap[std::make_pair(id, indx)] = port;
    }

    void addPort(const descriptor::Output &output, const IENetAPI::OutputPort &port)
    {
        size_t srcOpId = output.get_node()->get_instance_id();
        size_t srcIndex = output.get_index();
        m_portMap[std::make_pair(srcOpId, srcIndex)] = port;
    }

    IENetAPI::OutputPort getPort(size_t id, size_t indx) const
    {
        auto it = m_portMap.find(std::make_pair(id, indx));
        if (it ==  m_portMap.end())
            THROW("port of id (") << id << ", " << indx << ") was not created yet.";
        return it->second;
    }

    IENetAPI::OutputPort getPortFromOutput(const descriptor::Output &output)
    {
        size_t srcOpId = output.get_node()->get_instance_id();
        size_t srcIndex = output.get_index();
        return getPort(srcOpId, srcIndex);
    }

    IENetAPI::OutputPort getPortFromInput(const descriptor::Input &input)
    {
        return getPortFromOutput(input.get_output());
    }

    //maybe someone would wish to override this
    virtual void addLayer(const shared_ptr<Node> &op);

    void convert(const Function::Ptr &function, IENetAPI::IENetBuilder &doc);

};


#endif //NGRAPH_NGRAPHIEBRIDGE_H
