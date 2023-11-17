#pragma once
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>

using cv::Mat;

class RVM
{
public:
	RVM();
	~RVM();

	int set_background(Mat& input);

	int inference(Mat& input, Mat& output);

private:

	int m_inferenceWidth = 1080;

	int m_inferenceHeight = 1920;

	float m_downsampleRatio = 0.4;

	std::string m_modelPath = "rvm_resnet50_fp16.torchscript";

	Mat m_background;

	torch::Device* m_device;
	torch::jit::script::Module m_model;
	at::Tensor m_greenTensor;
	c10::optional<torch::Tensor> m_tensorRec0;
    c10::optional<torch::Tensor> m_tensorRec1;
    c10::optional<torch::Tensor> m_tensorRec2;
    c10::optional<torch::Tensor> m_tensorRec3;

};

