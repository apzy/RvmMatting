#include "RVM.h"

RVM::RVM()
{
	m_device = new torch::Device(torch::kCUDA);
	try
	{
		m_model = torch::jit::load(m_modelPath);
		m_model.to(*m_device);
		m_model.eval();
		std::cout << "finish load the model\n";
	}
	catch (const c10::Error& e)
	{
		std::cout << "error loading the model\n";
		std::cout << e.what() << std::endl;
	}
	m_greenTensor = torch::tensor({ 0.f / 255,255.f / 255,0.f / 255 }).toType(torch::kFloat16).to(*m_device).view({ 1, 3, 1, 1 });
}

RVM::~RVM()
{}

int RVM::set_background(Mat& input)
{
	m_background = input.clone();
	return 0;
}

int RVM::inference(Mat& input, Mat& output)
{
	Mat srcMat = input.clone();
	output = Mat(cv::Size(m_inferenceWidth, m_inferenceHeight), CV_8UC3);
	if (input.rows != m_inferenceHeight || input.cols != m_inferenceWidth)
	{
		cv::resize(input, srcMat, cv::Size(m_inferenceWidth, m_inferenceHeight));
	}
	at::Tensor srcTensor = torch::from_blob(srcMat.data, { m_inferenceHeight,m_inferenceWidth,3 }, torch::kByte);
	srcTensor = srcTensor.to(*m_device);
	srcTensor = srcTensor.permute({ 2, 0, 1 }).contiguous();
	srcTensor = srcTensor.to(torch::kFloat16).div(255);
	srcTensor.unsqueeze_(0);

	{
		torch::NoGradGuard no_grad;
		auto outputs = m_model.forward({ srcTensor, m_tensorRec0, m_tensorRec1, m_tensorRec2, m_tensorRec3, m_downsampleRatio }).toList();

		const auto& fgr = outputs.get(0).toTensor();
		const auto& pha = outputs.get(1).toTensor();
		m_tensorRec0 = outputs.get(2).toTensor();
		m_tensorRec1 = outputs.get(3).toTensor();
		m_tensorRec2 = outputs.get(4).toTensor();
		m_tensorRec3 = outputs.get(5).toTensor();
		auto res_tensor = (1 - pha) * m_greenTensor + pha * srcTensor;
		res_tensor = res_tensor.mul(255).permute({ 0, 2, 3, 1 })[0].to(torch::kU8).contiguous().cpu();
		std::memcpy((void*)output.data, res_tensor.data_ptr(), sizeof(torch::kU8) * res_tensor.numel());
	}

	return 0;
}
