# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You can obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        qwen36_kv_rope_scatter_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Qwen36KvRopeScatter
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE qwen36_kv_rope_scatter ACLNNTYPE aclnn)
endif()
