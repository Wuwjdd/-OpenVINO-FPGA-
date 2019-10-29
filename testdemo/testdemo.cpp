#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <chrono>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <utility>

#include<cmath>//CÓïÑÔÊÇmath.h



#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/ocv_common.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include "test_demo.h"

using namespace InferenceEngine;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
	// ---------------------------Parsing and validating the input arguments--------------------------------------
	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
	if (FLAGS_h) {
		showUsage();
		return false;
	}
	slog::info << "Parsing input parameters" << slog::endl;

	if (FLAGS_i.empty()) {
		throw std::logic_error("Parameter -i is not set");
	}

	if (FLAGS_m.empty()) {
		throw std::logic_error("Parameter -m is not set");
	}
	return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
	if (FLAGS_auto_resize) {
		/* Just set input blob containing read image. Resize and layout conversion will be done automatically */
		inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
	}
	else {
		/* Resize and copy data from the image to the input blob */
		Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
		matU8ToBlob<uint8_t>(frame, frameBlob);
	}
}

cv::Mat DarkChannl(cv::Mat &im, int sz) {

	std::vector<cv::Mat>mv(3);

	cv::split(im, mv);
	cv::Mat dc = cv::min(cv::min(mv[0], mv[1]), mv[2]);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(sz, sz));

	cv::Mat dark;
	cv::erode(dc, dark, kernel);

	return dark;

}

int* AtmLight(cv::Mat &im, cv::Mat &dark) {

	//std::vector<cv::Mat>mv(3);
	//cv::split(im,mv);
	size_t h = im.rows;
	size_t w = im.cols;
	size_t imsz = h * w;
	size_t num_P = imsz / 1000;

	int *arr = new int[3];
	arr[0] = 0;
	arr[1] = 0;
	arr[2] = 0;


	cv::Mat  m_min, m_max;
	double minVal, maxVal;
	int    minIdx[2] = {}, maxIdx[2] = {};

	for (size_t i = 0; i < num_P; i++) {
		cv::minMaxIdx(dark, &minVal, &maxVal, minIdx, maxIdx);
		for (int c = 0; c < 3; c++)
		{
			int pix_h = maxIdx[0];
			int pix_w = maxIdx[1];
			int pixl = im.at<cv::Vec3b>(pix_h, pix_w)[c];

			if (pixl > arr[c]) {
				arr[c] = pixl;
			}
			dark.at<uchar>(pix_h, pix_w) = 0;
		}
	}

	//slog::info << " arr =  " << arr << slog::endl;


	return arr;
}

cv::Mat Guidedfilter(cv::Mat &im, cv::Mat &p, int r, float eps) {

	cv::Mat mean_I, mean_p, mean_Ip, cov_Ip, mean_II, var_I, a, b, mean_a, mean_b, q;

	cv::boxFilter(im, mean_I, -1, cv::Size(r, r));
	cv::boxFilter(p, mean_p, -1, cv::Size(r, r));
	cv::boxFilter(im.mul(p), mean_Ip, -1, cv::Size(r, r));
	cv::boxFilter(im.mul(im), mean_II, -1, cv::Size(r, r));

	cov_Ip = mean_Ip - mean_I.mul(mean_p);

	var_I = mean_II - mean_I.mul(mean_I);
	a = cov_Ip / (var_I + eps);
	b = mean_p - a.mul(mean_I);

	cv::boxFilter(a, mean_a, -1, cv::Size(r, r));
	cv::boxFilter(b, mean_b, -1, cv::Size(r, r));

	q = mean_a.mul(im) + mean_b;

	return q;
}

cv::Mat TransmissionRefine(cv::Mat &im, cv::Mat &et) {

	cv::Mat et1, gray, gray1, dst;

	cv::cvtColor(im, gray, CV_BGR2GRAY);
	et.convertTo(et1, CV_64FC1);
	gray.convertTo(gray1, CV_64FC1);
	//slog::info << "sizze: " << gray.type() << slog::endl; 
	gray1 = gray1 / 255.0;

	int r = 60;
	float eps = 0.0001;
	dst = Guidedfilter(gray1, et1, r, eps);

	return dst;
}

cv::Mat Recover(cv::Mat &im, cv::Mat &t, int b, int g, int r) {
	int *A = new int[3];
	A[0] = b;
	A[1] = g;
	A[2] = r;

	cv::Mat rec = cv::Mat::zeros(im.rows, im.cols, CV_64FC3);
	cv::Mat im1;
	std::vector<cv::Mat> bgr;

	t = max(t, 25.5);
	im.convertTo(im1, CV_64FC3);
	//slog::info << "zhelim00 " << slog::endl;  
	cv::split(im1, bgr);


//slog::info << "zhelima " << test.type() << slog::endl; 
	for (int i = 0; i < 3; i++) {
		bgr.at(i) = (bgr.at(i) - A[i]) / t * 255 + A[i];
	}
	//slog::info << "zhelima " << test.type() << slog::endl; 
	cv::merge(bgr, rec);
	//slog::info << "zhelima " << test.type() << slog::endl; 
	return rec;
}


int main(int argc, char *argv[]) {
	//////////////////////////////////////////////////////////////////////////////
	//žùŸÝÊÓÆµ³ßŽçÐÞžÄSIZE
	cv::VideoWriter writer("VideoTest.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),12.0, cv::Size(640, 360));
	/////////////////////////////////////////////////////////////////////////////
	try {
		slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

		// ------------------------------ Parsing and validation of input args ---------------------------------
		if (!ParseAndCheckCommandLine(argc, argv)) {
			return 0;
		}

		/** This vector stores paths to the processed images **/
		cv::VideoCapture cap;
		if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
			throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
		}

		//int delay = 33;
		//double inferenceTime = 0.0;

		cv::Mat frame;  cap >> frame;
		cv::Mat next_frame;

		if (!cap.read(frame)) {
			throw std::logic_error("Failed to get frame from cv::VideoCapture");
		}
		//estimator.estimate(image);  // Do not measure network reshape, if it happened
		if (!FLAGS_no_show) {
			std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
		}
		// -----------------------------------------------------------------------------------------------------

		// --------------------------- 1. Load Plugin for inference engine -------------------------------------
		slog::info << "Loading plugin" << slog::endl;
		InferencePlugin plugin = PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);

		/** Loading default extensions **/
		if (FLAGS_d.find("CPU") != std::string::npos) {
			/**
			 * cpu_extensions library is compiled from "extension" folder containing
			 * custom MKLDNNPlugin layer implementations. These layers are not supported
			 * by mkldnn, but they can be useful for inferring custom topologies.
			**/
			plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
		}

		if (!FLAGS_l.empty()) {
			// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
			IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
			plugin.AddExtension(extension_ptr);
			slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
		}
		if (!FLAGS_c.empty()) {
			// clDNN Extensions are loaded from an .xml description and OpenCL kernel files
			plugin.SetConfig({ {PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c} });
			slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
		}

		/** Setting plugin parameter for per layer metrics **/
		if (FLAGS_pc) {
			plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
		}
		/** Printing plugin version **/
		//printPluginVersion(plugin, std::cout);
		// -----------------------------------------------------------------------------------------------------

		// --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
		//slog::info << "Loading network files" << slog::endl;


		CNNNetReader networkReader;
		/** Read network model **/
		networkReader.ReadNetwork(FLAGS_m);

		/** Extract model name and load weights **/
		std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
		networkReader.ReadWeights(binFileName);
		CNNNetwork network = networkReader.getNetwork();
		// -----------------------------------------------------------------------------------------------------


		slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
		InputsDataMap inputInfo(network.getInputsInfo());
		if (inputInfo.size() != 1) {
			throw std::logic_error("This demo accepts networks that have only one input");
		}
		InputInfo::Ptr& input = inputInfo.begin()->second;
		auto inputName = inputInfo.begin()->first;
		input->setPrecision(Precision::U8);
		if (FLAGS_auto_resize) {
			input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
			input->getInputData()->setLayout(Layout::NHWC);
		}
		else {
			input->getInputData()->setLayout(Layout::NCHW);
		}


		// --------------------------- Prepare output blobs ----------------------------------------------------
		std::cout << "[ INFO ] Checking Person Detection outputs" << std::endl;
		OutputsDataMap outputInfo(network.getOutputsInfo());
		//std::string outputName;
		if (outputInfo.size() != 1) {
			throw std::logic_error("Person Detection network should have only one output");
		}
		DataPtr& output = outputInfo.begin()->second;
		auto outputName = outputInfo.begin()->first;
		const SizeVector outputDims = output->getTensorDesc().getDims();
		output->setPrecision(Precision::FP32);
		output->setLayout(Layout::NCHW);


		//size_t N = outputDims[0];
		//size_t C = outputDims[1];
		size_t H = outputDims[2];
		size_t W = outputDims[3];
		//size_t image_stride = W * H * C;

		size_t r_H = frame.rows;
		size_t r_W = frame.cols;


		// --------------------------- 4. Loading model to the plugin ------------------------------------------
		slog::info << "Loading model to the plugin" << slog::endl;
		ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});

		// --------------------------- 5. Create infer request -------------------------------------------------
		//InferRequest infer_request = executable_network.CreateInferRequest();
		InferRequest::Ptr infer_request_next = executable_network.CreateInferRequestPtr();
		InferRequest::Ptr infer_request_curr = executable_network.CreateInferRequestPtr();

		// --------------------------- 6. Doing inference ------------------------------------------------------
		slog::info << "Start inference " << slog::endl;

		bool isLastFrame = false;
		bool isFirstFrame = true;  // execution is always started using SYNC mode
		//bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

		typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
		//auto total_t0 = std::chrono::high_resolution_clock::now();
		//auto wallclock = std::chrono::high_resolution_clock::now();
		//double ocv_decode_time = 0, ocv_render_time = 0;

		std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
		while (true) {
			auto t0 = std::chrono::high_resolution_clock::now();
			// Here is the first asynchronous point:
			// in the Async mode, we capture frame to populate the NEXT infer request
			// in the regular mode, we capture frame to the CURRENT infer request
			if (!cap.read(next_frame)) {
				if (next_frame.empty()) {
					isLastFrame = true;  // end of video file
				}
				else {
					throw std::logic_error("Failed to get frame from cv::VideoCapture");
				}
			}

			if (!isLastFrame) {
				FrameToBlob(next_frame, infer_request_next, inputName);
			}

//typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
			auto t1 = std::chrono::high_resolution_clock::now();
			//ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

			t0 = std::chrono::high_resolution_clock::now();
			// Main sync point:
			// in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
			// in the regular mode, we start the CURRENT request and wait for its completion
			if (isFirstFrame) {
				infer_request_curr->StartAsync();
				infer_request_next->StartAsync();
				isFirstFrame = false;
			}
			else {
				if (!isLastFrame) {
					infer_request_next->StartAsync();
				}
			}

			

			cv::Mat dst, dst1;
			if (OK == infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
				auto t0 = std::chrono::high_resolution_clock::now();
				
				const Blob::Ptr outputblob = infer_request_curr->GetBlob(outputName);
				const auto output_data = outputblob->buffer().as<float*>();
				cv::Mat outblob(H, W, CV_32FC1, cv::Scalar(0));
				/** Iterating over each pixel **/

				for (size_t w = 0; w < W; w++) {
					for (size_t h = 0; h < H; h++) {
						outblob.at<float>(h, w) = fabs(output_data[W * h + w] * 400000);
						//slog::info << "output:  " << std::to_string(outblob.at<float>(h, w)) << slog::endl;
					}
				}
				//for (size_t m = 0; m < image_stride;m++)
				//{
					//slog::info << "output:  "<< std::to_string(output_data[m]) << slog::endl;
				//}

				//slog::info << "##############output:"<< std::to_string(dst.type()) << slog::endl;
				//std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@" << outblob.type() << std::endl;

				cv::Mat resimg = frame.clone();
				cv::Mat outblob1, Transmission, dark;
				cv::resize(outblob, outblob1, cv::Size(r_W, r_H), 0, 0, cv::INTER_LINEAR);
				Transmission = TransmissionRefine(resimg, outblob1);
				Transmission = 255 - Transmission;
				//dst = Transmission;
				dark = DarkChannl(resimg, 15);
				//dst = dark;
				int *arr = AtmLight(resimg, dark);
				//slog::info << "A: " << arr[0] << arr[1] << arr[2] << slog::endl;
				dst = Recover(resimg, Transmission, arr[0], arr[1 ], arr[2]);
				//dst = outblob;
				//cv::imshow("defog", dst);
				//std::cout << "############################" << dst.type() << std::endl;



				auto t1 = std::chrono::high_resolution_clock::now();
				ms dtime = std::chrono::duration_cast<ms>(t1 - t0);
				std::ostringstream out;
				out << " time  : " << std::fixed << std::setprecision(2) << dtime.count()
					<< " ms (" << 1000.f / dtime.count() << " fps)";
				
				
				
				
				double Min = 0.0;
				double Max = 0.0;

				std::vector<cv::Mat>SrcMatpart(3);
				//Éú³ÉÓëÍšµÀÊýÊýÄ¿ÏàµÈµÄÍŒÏñÈÝÆ÷
				cv::split(dst, SrcMatpart);
				cv::Mat temp;
				cv::merge(SrcMatpart, temp);

				for (size_t w = 0; w < r_W; w++) {
					for (size_t h = 0; h < r_H; h++) {
						if (dst.at<double>(h, w) > Max) {
							Max = dst.at<double>(h, w);
						}
						if (dst.at<double>(h, w) < Min) {
							Min = dst.at<double>(h, w);
						}
						//slog::info << "output:  " << std::to_string(outblob.at<float>(h, w)) << slog::endl;
					}
				}
				dst.convertTo(dst1, CV_8U, 255.0 / (Max - Min), -255.0*Min / (Max - Min));

				putText(dst1, out.str(), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2, 8);
				writer << dst1;
				//cv::imwrite("defog_video.png", dst1);
				cv::imshow("defog_video", dst1);
				cv::imshow("fog_video", frame);
               
			
			}


			t1 = std::chrono::high_resolution_clock::now();
			//ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

			if (isLastFrame) {
				break;
			}


			next_frame.copyTo(frame);

			infer_request_curr.swap(infer_request_next);

			const int key = cv::waitKey(1);
			if (27 == key)  // Esc
				break;

		}

		//if (FLAGS_pc) {
			//printPerformanceCounts(*infer_request_curr, std::cout);
		//}

	}
	catch (const std::exception& error) {
		std::cerr << "[ ERROR ] " << error.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
		return 1;
	}

	slog::info << "Execution successful" << slog::endl;
	return 0;
}
