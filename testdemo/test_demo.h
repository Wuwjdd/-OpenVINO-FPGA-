// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to an .bmp image.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Optional. Path to a plugin folder.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";\

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. " \
"The demo will look for a suitable plugin for a specified device. Default value is CPU.";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. Default value is 1";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to the .xml file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementations.";

static const char no_show_processed_video[] = "Optional. No show processed video.";

static const char input_resizable_message[] = "Optional. Enable resizable input with support of ROI crop and auto resize.";

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);

/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Iterations count (default 1)
DEFINE_uint32(ni, 1, iterations_count_message);

DEFINE_bool(no_show, false, no_show_processed_video);

DEFINE_bool(auto_resize, false, input_resizable_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "segmentation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"               " << image_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -pp \"<path>\"              " << plugin_path_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -ni \"<integer>\"           " << iterations_count_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
	std::cout << "    -no_show                     " << no_show_processed_video << std::endl;
	std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
}
