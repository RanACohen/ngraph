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
#include "ie_net_builder.hpp"
#include "file_utils.h"
#include <cassert>
#include <vector>
#include <string>
#include <memory>

namespace IENetAPI {

    extern int layer_name_count;

    inline OutputPort addOutput(const IELayer &layer, const InferenceEngine::SizeVector &dims) {
        std::string d_name = layer->name;
        if (!layer->outData.empty()) {
            std::stringstream oss;
            oss << d_name << ":" << layer->outData.size();
            d_name = oss.str();
        }
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32,dims, InferenceEngine::Layout::ANY);
        auto data = std::make_shared<InferenceEngine::Data>(d_name,td);
        layer->outData.push_back(data);
        data->creatorLayer = layer;
        return data;
    }


    template<typename T>
    void addAttr(IELayer layer, const std::string &a_name, T val) {
        std::stringstream oss;
        oss << val;
        layer->params[a_name] = oss.str();
    }

    template<typename T, typename S>
    std::shared_ptr<T> As(const std::shared_ptr<S> &src) {
        return std::dynamic_pointer_cast<T>(src);
    }


    /*
    * @brief Just creates a generic layer given a type
    */
    inline IELayer Generic(const std::string &type, InferenceEngine::Precision::ePrecision p = InferenceEngine::Precision::FP32) {
        std::string name = type + "-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prms;
        prms.precision = p;
        prms.name = name;
        prms.type = type;
        auto layer = std::make_shared<InferenceEngine::CNNLayer>(prms);
        return layer;
    }

    /*
    * @brief Creates a generic layer with one input and one output
    */
    inline IELayer Generic(const std::string &type, const OutputPort &src) {
        auto srcOwner = src->creatorLayer;
        InferenceEngine::Precision::ePrecision p;
        if (!srcOwner.expired())
            p = srcOwner.lock()->precision;
        else
            p = src->precision;
        auto layer = Generic(type, p);
        src >> layer;
        addOutput(layer, src->getDims());
        return layer;
    }

    inline OutputPort output(const IELayer &src, int index = 0) {
        return src->outData[index];
    }

    inline void addInput(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::DataPtr &src)
    {
        src >> layer;
    }

    inline IELayer LayerOf(const OutputPort &src) {
        return src->creatorLayer.lock();
    }

    inline IELayer Generic(const std::string &type, const IELayer &src) {
        return Generic(type, output(src));
    }

    template<typename T, typename A>
    std::string dumpVec(std::vector<T, A> const &vec) {
        if (vec.empty()) return "[]";
        std::stringstream oss;
        oss << "[" << vec[0];
        for (size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
        oss << "]";
        return oss.str();
    }

namespace SumLayer {
    static IELayer create(const OutputPort &src1, const OutputPort &src2) {
        std::string name = "Sum-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto sum = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
        sum->type = "Eltwise";
        src1 >> sum;
        src2 >> sum;
        if (src1->getDims() != src2->getDims())
            THROW_IE_EXCEPTION << "input sizes for Element wise Sum do not match: "
                               << src1->getDims() << " and " << src2->getDims();
        addOutput(sum, src1->getDims());
        return sum;
    }
};  // namespace SumLayer

    inline OutputPort operator+(const OutputPort &a, const OutputPort &b) {
        return output(SumLayer::create(a, b));
    }


    static OutputPort ScaleShiftNode(const OutputPort &src, const IETensor::Ptr &scale, const IETensor::Ptr &bias) {
        std::string name = "ConstMul-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        prm.type = "ScaleShift";
        auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);

        src >> l;
        l->_weights = scale;
        l->_broadcast = false;
        l->_biases = bias;
        l->blobs["biases"] = bias;
        return addOutput(l, src->getDims());
    }

namespace FCLayer {

    static IELayer create(const IETensor::Ptr &weights, const OutputPort &src) {
        std::string name = "FC-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = weights->precision(); // get layer precision from the weights
        prm.name = name;

        auto inDims = src->getDims();  // (batch, IFM)

        auto wDim = weights->getTensorDesc().getDims();

        IR_ASSERT(inDims.size() == 2);

        unsigned int ofm = 0;
        if (wDim.size() == 2) {
            IR_ASSERT(inDims[1] == wDim[1]);  // Weights: (Out,In)
            ofm = static_cast<unsigned int>(wDim[0]);  // Out
        } else if (wDim.size() == 1) {  // linear, just a blob, line in IR
            ofm = static_cast<unsigned int>(weights->size() / inDims[1]);
            IR_ASSERT(inDims[1] * ofm == weights->size());  // should be divided properly
        } else {
            THROW_IE_EXCEPTION << "expecting weights for FC only as 1 dim (blob) or 2 dim (Matrix)";
        }

        auto fc = std::make_shared<InferenceEngine::FullyConnectedLayer>(prm);
        fc->type = "FullyConnected";

        fc->_out_num = ofm;
        // todo: assert that input should be cols
        addOutput(fc, { inDims[0], static_cast<uint32_t>(fc->_out_num) });
        src >> fc;
        fc->_weights = weights;
        fc->blobs["weights"] = weights;  // todo: have setter for those layers...
        return fc;
    }
};  // namespace FCLayer


    inline InferenceEngine::CNNLayer::Ptr operator*(const IETensor::Ptr &weights, const IELayer &b) {
        return FCLayer::create(weights, output(b));
    }

    inline OutputPort operator*(const IETensor::Ptr &weights, const OutputPort &op) {
        return output(FCLayer::create(weights, op));
    }

    inline IELayer AddBiases(const IELayer &lhs, const IETensor::Ptr &biases) {
        auto fc = As<InferenceEngine::WeightableLayer>(lhs);
        if (fc) {
            // todo: check if biases was not lready being set
            fc->_biases = biases;
            fc->blobs["biases"] = biases;
            return lhs;  // it was fused with prev layer
        } else {
            // need to create an add with Const here using ScaleShift with no weights...
            THROW_IE_EXCEPTION << "not implemented yet";
        }
    }

    inline OutputPort CreateConst(IENetBuilder &doc, const IETensor::Ptr &constValue) {
        // use const layer with element wise add
        auto constNode = Generic("Const");
        doc.add(constNode);
        constNode->blobs["custom"] = constValue;
        return addOutput(constNode, constValue->getTensorDesc().getDims());
    }


    inline OutputPort AddConst(IENetBuilder &doc, const OutputPort &src, const IETensor::Ptr &biases) {
        // this depends on the plugin, see E-mail
        bool useScaleShift = false;

        if (useScaleShift) {
            return ScaleShiftNode(src, nullptr, biases);
        }
        // use const layer with element wise add
        return src + CreateConst(doc, biases);
    }

    inline OutputPort operator+(const OutputPort &src, const IETensor::Ptr &biases) {
        auto l = LayerOf(src);
        return output(AddBiases(l, biases));
    }

namespace ConvLayer {

    static IELayer create(const OutputPort &src) {
        std::string name = "Conv-";  // todo: make it unique
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        name = name << layer_name_count++;
        prm.name = name;
        auto conv_layer = std::make_shared<InferenceEngine::ConvolutionLayer>(prm);
        conv_layer->type = "Convolution";
        src >> conv_layer;
        return conv_layer;
    }
};  // namespace ConvLayer

struct Point2D {
        int x, y;

        inline int size() const {
            return x * y;
        }
};

    inline Point2D operator+(const Point2D &a, const Point2D &b) {
        return{ a.x + b.x, a.y + b.y };
    }

    inline Point2D operator-(const Point2D &a, const Point2D &b) {
        return{ a.x - b.x, a.y - b.y };
    }

    inline Point2D operator*(const Point2D &a, const Point2D &b) {
        return{ a.x * b.x, a.y * b.y };
    }

    inline Point2D operator/(const Point2D &a, const Point2D &b) {
        return{ a.x / b.x, a.y / b.y };
    }

    inline Point2D operator+(const Point2D &a, const int &rhs) {
        return{ a.x + rhs, a.y + rhs };
    }

struct ConvolutionParams {
        int groups = 1;
        Point2D kernel, stride = { 1 }, pad_start = { 0 }, pad_end = { 0 };
        int num_output_planes;
        IETensor::Ptr weights;
};

    inline size_t n(const OutputPort &src) {
        auto dims = src->getDims();
        return dims.size() == 4 ? dims[1] : dims[2];
    }

    inline OutputPort Convolution(const OutputPort &src, const ConvolutionParams &prms) {
        auto ret = As<InferenceEngine::ConvolutionLayer>(ConvLayer::create(src));
        auto inDims = src->getDims();
        IR_ASSERT(inDims.size() == 4);
        IR_ASSERT(prms.kernel.size() * n(src) * prms.num_output_planes == prms.weights->size());

        ret->_weights = prms.weights;
        ret->blobs["weights"] = prms.weights;
        ret->_kernel_x = prms.kernel.x;
        ret->_kernel_y = prms.kernel.y;
        ret->_stride_x = prms.stride.x;
        ret->_stride_y = prms.stride.y;
        ret->_padding_x = prms.pad_start.x;
        ret->_padding_y = prms.pad_start.y;
        ret->_dilation_x = 1;
        ret->_dilation_y = 1;

        ret->_group = prms.groups;
        ret->_out_depth = prms.num_output_planes;
        ret->params["group"] = std::to_string(ret->_group);
        ret->params["kernel-x"] = std::to_string(ret->_kernel_x);
        ret->params["kernel-y"] = std::to_string(ret->_kernel_y);
        ret->params["output"] = std::to_string(ret->_out_depth);
        ret->params["pad-x"] = std::to_string(ret->_padding_x);
        ret->params["pad-y"] = std::to_string(ret->_padding_y);
        ret->params["stride-x"] = std::to_string(ret->_stride_x);
        ret->params["stride-y"] = std::to_string(ret->_stride_y);

        Point2D in_size = { static_cast<int>(inDims[3]), static_cast<int>(inDims[2]) };
        // todo: handle uneven padding
        Point2D out_size = (in_size + prms.pad_start + prms.pad_end - prms.kernel) / prms.stride + 1;
        addOutput(ret, { inDims[0], (size_t)prms.num_output_planes, (size_t)out_size.y, (size_t)out_size.x });
        return output(ret);
    }

struct BatchNormParams {
        float epsilon;
        IETensor::Ptr weights;
        IETensor::Ptr bias;
};

    inline IELayer BatchNormalization(const OutputPort &src, BatchNormParams &prms) {
        auto inp = src;
        std::string name = "BatchNormalization-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto l = std::make_shared<InferenceEngine::BatchNormalizationLayer>(prm);
        l->type = "BatchNormalization";
        src >> l;
        l->epsilon = prms.epsilon;
        l->_weights = prms.weights;
        l->_biases = prms.bias;
        addOutput(l, inp->getDims());
        return l;
    }

    inline OutputPort LRN(const OutputPort &src, float alpha, float beta, int local_size, bool isAcross = true, float k = 1) {
        auto inp = src;
        std::string name = "Norm-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto l = std::make_shared<InferenceEngine::NormLayer>(prm);
        l->type = "Norm";

        src >> l;
        l->_alpha = alpha;
        l->_beta = beta;
        l->_isAcrossMaps = isAcross;
        l->_size = local_size;
        l->_k = (unsigned int)k;
        return addOutput(l, inp->getDims());
    }

    inline OutputPort Crop(const OutputPort &src,
        const std::vector<int> &axis,
        const std::vector<int> &dim,
        const std::vector<int> &offset) {
        auto inp = src;
        std::string name = "Crop-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto l = std::make_shared<InferenceEngine::CropLayer>(prm);
        l->type = "Crop";
        src >> l;
        l->axis = axis;
        l->dim = dim;
        l->offset = offset;
        InferenceEngine::SizeVector sv(dim.begin(), dim.end());
        return addOutput(l, sv);
    }

    inline OutputPort Pooling(const OutputPort &inp,
        const Point2D &kernel,
        const Point2D &stride,
        const Point2D &pad,
        InferenceEngine::PoolingLayer::PoolType type) {
        auto src = inp;
        std::string name = "Pooling-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto ret = std::make_shared<InferenceEngine::PoolingLayer>(prm);
        ret->type = "Pooling";
        ret->_kernel_x = kernel.x;
        ret->_kernel_y = kernel.y;
        ret->_stride_x = stride.x;
        ret->_stride_y = stride.y;
        ret->_padding_x = pad.x;
        ret->_padding_y = pad.y;
        ret->_type = type;
        ret->_exclude_pad = true;

        auto inDims = src->getDims();

        Point2D in_size = { static_cast<int>(inDims[3]), static_cast<int>(inDims[2]) };
        // todo: handle uneven padding
        Point2D out_size = (in_size + pad + pad - kernel + stride + -1) / stride + 1;  // add stride-1 to round ceiling
        src >> ret;
        return addOutput(ret, { inDims[0], inDims[1], (size_t)out_size.y, (size_t)out_size.x });
    }

    inline OutputPort Pooling(const OutputPort &inp,
        const Point2D &kernel,
        const Point2D &stride,
        const Point2D &pad_start,
        const Point2D &pad_end,
        InferenceEngine::PoolingLayer::PoolType type) {
        auto src = inp;
        std::string name = "Pooling-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto ret = std::make_shared<InferenceEngine::PoolingLayer>(prm);
        ret->type = "Pooling";
        ret->_kernel_x = kernel.x;
        ret->_kernel_y = kernel.y;
        ret->_stride_x = stride.x;
        ret->_stride_y = stride.y;
        ret->_padding_x = pad_start.x;
        ret->_padding_y = pad_start.y;
        ret->_type = type;
        ret->_exclude_pad = true;

        auto inDims = src->getDims();

        Point2D in_size = { static_cast<int>(inDims[3]), static_cast<int>(inDims[2]) };
        // todo: handle uneven padding
        Point2D out_size = (in_size + pad_start + pad_end - kernel + stride + -1) / stride + 1;  // add stride-1 to round ceiling
        src >> ret;
        return addOutput(ret, { inDims[0], inDims[1], (size_t)out_size.y, (size_t)out_size.x });
    }


namespace MulLayer {

        static IELayer create(const OutputPort &src1, const OutputPort &src2) {
            std::string name = "Mul-";  // todo: make it unique
            name = name << layer_name_count++;
            InferenceEngine::LayerParams prm;
            prm.precision = InferenceEngine::Precision::FP32;
            prm.name = name;
            auto mul = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
            mul->type = "Mul";
            mul->_operation = InferenceEngine::EltwiseLayer::Prod;
            src1 >> mul;
            src2 >> mul;
            if (src1->getDims() != src2->getDims()) THROW_IE_EXCEPTION << "input sizes for Element wise Mul do not match";
            addOutput(mul, src1->getDims());
            return mul;
        }
};  // namespace MulLayer

    inline OutputPort operator*(const OutputPort &a, const OutputPort &b) {
        return output(MulLayer::create(a, b));
    }

namespace ScaleShift {

        static OutputPort Diagnoal(const Vector &weights, const OutputPort &src) {
            std::string name = "ConstMul-";  // todo: make it unique
            name = name << layer_name_count++;
            InferenceEngine::LayerParams prm;
            prm.precision = InferenceEngine::Precision::FP32;
            prm.name = name;
            auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
            l->type = "ConstMul";
            src >> l;
            addOutput(l, src->getDims());
            l->_weights = weights.data;
            if (weights.length == 1) {
                l->_broadcast = 0;
            } else if (weights.length == src->getDims()[1]) {
                l->_broadcast = 1;
            }

            return output(l);
        }

        static InferenceEngine::CNNLayer::Ptr create(OutputPort src,
            IETensor::Ptr scale,
            IETensor::Ptr bias) {
            std::string name = "ConstMul-";  // todo: make it unique
            name = name << layer_name_count++;
            InferenceEngine::LayerParams prm;
            prm.precision = InferenceEngine::Precision::FP32;
            prm.name = name;
            auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
            l->type = "ScaleShift";
            src >> l;
            l->_weights = scale;
            l->_broadcast = false;
            addOutput(l, src->getDims());
            AddBiases(l, bias);
            return l;
        }
};  // namespace ScaleShift

    inline OutputPort operator*(const Vector &weights, const IELayer &b) {
        return (ScaleShift::Diagnoal(weights, output(b)));
    }

    inline OutputPort operator*(const Vector &weights, const OutputPort &op) {
        return (ScaleShift::Diagnoal(weights, op));
    }

namespace ActivationLayer {
        extern const std::string Sigmoid;

        extern const std::string Tanh;

        extern const std::string ReLU;

        static IELayer create(const OutputPort &src, const std::string &type) {
            std::string name = type + "-";  // todo: make it unique
            name = name << layer_name_count++;
            IELayer layer;
            if ((strncasecmp(type.c_str(), "relu", type.size()) == 0)) {
                layer = Generic("ReLU", src);
            } else {
                layer = Generic("Activation", src);
                addAttr(layer, "type", type);
            }
            addOutput(layer, src->getDims());
            return layer;
        }

        static IELayer create(const IELayer &src, const std::string &type) {
            return create(output(src), type);
        }
};  // namespace ActivationLayer

    template<typename T>
    OutputPort ReLU(const T &src) {
        return output(ActivationLayer::create(src, ActivationLayer::ReLU));
    }

    template<typename T>
    OutputPort Sigmoid(const T &src) {
        return output(ActivationLayer::create(src, ActivationLayer::Sigmoid));
    }

    template<typename T>
    OutputPort Tanh(const T &src) {
        return output(ActivationLayer::create(src, ActivationLayer::Tanh));
    }

namespace SplitUtil {

        static IELayer create(int size, const OutputPort &src, int axis = 1) {
            std::string name = "Split-";  // todo: make it unique
            name = name << layer_name_count++;
            InferenceEngine::LayerParams prm;
            prm.precision = InferenceEngine::Precision::FP32;
            prm.name = name;
            auto me = std::make_shared<InferenceEngine::SplitLayer>(prm);
            me->type = "Split";
            addAttr(me, "axis", axis);
            src >> me;
            auto out_dim = src->getDims();
            // axis = static_cast<int>(out_dim.size()) - axis - 1;  // todo: we are all in reverse here :-(
            out_dim[axis] = out_dim[axis] / size;
            IR_ASSERT(out_dim[axis] * size == src->getDims()[axis]);

            for (int i = 0; i < size; i++) {
                addOutput(me, out_dim);
            }
            return me;
        }
};  // namespace SplitUtil

    inline std::vector<OutputPort> Split(const OutputPort &src, int splitElements, int axis = 1) {
        return SplitUtil::create(splitElements, src, axis)->outData;
    }

    inline std::vector<OutputPort> Split(const IELayer &src, int splitElements, int axis = 1) {
        return Split(output(src), splitElements, axis);
    }

    inline OutputPort Concat(const std::vector<OutputPort> inputs, int axis = 1) {
        std::string name = "Concat-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto ret = std::make_shared<InferenceEngine::ConcatLayer>(prm);
        ret->type = "Concat";
        addAttr(ret, "axis", axis);
        inputs[0] >> ret;
        auto outDim = inputs[0]->getDims();

        // it was fixed, should be backward compatiobale though...
        // axis = static_cast<int>(outDim.size()) - axis - 1;  // todo: we are all in reverse here :-(
        auto axisSize = outDim[axis];
        for (int i = 1; i < inputs.size(); ++i) {
            inputs[i] >> ret;
            axisSize += inputs[i]->getDims()[axis];
        }
        outDim[axis] = axisSize;
        return addOutput(ret, outDim);
    }

    // template<typename T>
    inline OutputPort Clamp(const OutputPort &src, float min, float max) {
        auto layer = Generic("Clamp", src);
        addAttr(layer, "min", min);
        addAttr(layer, "max", max);
        return output(layer);
    }

    inline OutputPort L2Normalization(const OutputPort &src, bool isAcross, bool isShareChannel) {
        auto layer = Generic("Normalize", src);
        addAttr(layer, "across_spatial", isAcross ? 1 : 0);
        addAttr(layer, "channel_shared", isShareChannel ? 1 : 0);
        return output(layer);
    }

    inline OutputPort Reshape(const TensorDims &newDims, const OutputPort &src) {
        if (sizeOf(src->getDims()) != sizeOf(newDims)) THROW("Cannot reorder different volumes");
        if (src->creatorLayer.lock()->type == "Reshape") {  // fuse reshapes
            src->setDims(newDims);
            return src;
        }

        auto op = output(Generic("Reshape", src));
        op->setDims(newDims);
        return op;
    }

    static OutputPort Softmax(const OutputPort &src) {
        std::string name = "Softmax-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto l = std::make_shared<InferenceEngine::SoftMaxLayer>(prm);
        l->type = "SoftMax";
        src >> l;
        addOutput(l, src->getDims());
        return output(l);
    }

    inline OutputPort Gather(const std::vector<OutputPort> inputs, int axis = 1) {
        std::string name = "Gather-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = InferenceEngine::Precision::FP32;
        prm.name = name;
        auto ret = std::make_shared<InferenceEngine::GenericLayer>(prm);
        ret->type = "Gather";
        addAttr(ret, "axis", axis);
        inputs[0] >> ret;
        inputs[1] >> ret;
        auto outDim = inputs[0]->getDims();
        //   axis = static_cast<int>(outDim.size()) - axis - 1;  // todo: we are all in reverse here :-(
        outDim[0] = inputs[1]->getDims()[1];
        addOutput(ret, outDim);
        return output(ret);
    }
}  // namespace IRBuilder
