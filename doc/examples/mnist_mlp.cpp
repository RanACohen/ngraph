/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdio>
#include <memory>
#include <string>

#include <ngraph/ngraph.hpp>

#include "layers.hpp"
#include "mnist.hpp"

using ngraph;

class MLP
{
};

std::shared_ptr<Function> make_mlp_function(const std::vector<size_t>& sizes)

    int main(int argc, const char* argv[])
{
    MNistDataLoader test_loader{128, MNistImageLoader::TEST, MNistLabelLoader::TEST};
    test_loader.open();

    return 0;
}
