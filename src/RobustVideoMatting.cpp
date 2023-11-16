#include "RobustVideoMatting.h"

RobustVideoMatting::RobustVideoMatting(string model_path)
{
	wstring widestr = wstring(model_path.begin(), model_path.end());
	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
	session_ = new Session(env, widestr.c_str(), sessionOptions);
}

void RobustVideoMatting::normalize_(Mat img, vector<float>& output)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];   ///BGR2RGB
				output[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

int64_t RobustVideoMatting::value_size_of(const std::vector<int64_t>& dims)
{
	if (dims.empty()) return 0;
	int64_t value_size = 1;
	for (const auto& size : dims) value_size *= size;
	return value_size;
}

vector<Ort::Value> RobustVideoMatting::transform(const Mat& mat)
{
	Mat src = mat.clone();
	const unsigned int img_height = mat.rows;
	const unsigned int img_width = mat.cols;
	vector<int64_t>& src_dims = dynamic_input_node_dims.at(0); // (1,3,h,w)
	src_dims.at(2) = img_height;
	src_dims.at(3) = img_width;

	// assume that rxi's dims and value_handler was updated by last step in a while loop.
	std::vector<int64_t>& r1i_dims = dynamic_input_node_dims.at(1); // (1,?,?h,?w)
	std::vector<int64_t>& r2i_dims = dynamic_input_node_dims.at(2); // (1,?,?h,?w)
	std::vector<int64_t>& r3i_dims = dynamic_input_node_dims.at(3); // (1,?,?h,?w)
	std::vector<int64_t>& r4i_dims = dynamic_input_node_dims.at(4); // (1,?,?h,?w)
	std::vector<int64_t>& dsr_dims = dynamic_input_node_dims.at(5); // (1)

	int64_t src_value_size = this->value_size_of(src_dims); // (1*3*h*w)
	int64_t r1i_value_size = this->value_size_of(r1i_dims); // (1*?*?h*?w)
	int64_t r2i_value_size = this->value_size_of(r2i_dims); // (1*?*?h*?w)
	int64_t r3i_value_size = this->value_size_of(r3i_dims); // (1*?*?h*?w)
	int64_t r4i_value_size = this->value_size_of(r4i_dims); // (1*?*?h*?w)
	int64_t dsr_value_size = this->value_size_of(dsr_dims); // 1

	dynamic_src_value_handler.resize(src_value_size);
	this->normalize_(src, dynamic_src_value_handler);
	std::vector<Ort::Value> input_tensors;
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_src_value_handler.data(), dynamic_src_value_handler.size(), src_dims.data(), src_dims.size()));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_r1i_value_handler.data(), r1i_value_size, r1i_dims.data(), r1i_dims.size()));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_r2i_value_handler.data(), r2i_value_size, r2i_dims.data(), r2i_dims.size()));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_r3i_value_handler.data(), r3i_value_size, r3i_dims.data(), r3i_dims.size()));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_r4i_value_handler.data(), r4i_value_size, r4i_dims.data(), r4i_dims.size()));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_dsr_value_handler.data(), dsr_value_size, dsr_dims.data(), dsr_dims.size()));
	return input_tensors;
}

void RobustVideoMatting::generate_matting(std::vector<Ort::Value>& output_tensors, MattingContent& content)
{
	Ort::Value& fgr = output_tensors.at(0); // fgr (1,3,h,w) 0.~1.
	Ort::Value& pha = output_tensors.at(1); // pha (1,1,h,w) 0.~1.
	auto fgr_dims = fgr.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int height = fgr_dims.at(2); // output height
	const unsigned int width = fgr_dims.at(3); // output width
	const unsigned int channel_step = height * width;
	// merge -> assign & channel transpose(CHW->HWC).
	float* fgr_ptr = fgr.GetTensorMutableData<float>();
	float* pha_ptr = pha.GetTensorMutableData<float>();
	Mat rmat(height, width, CV_32FC1, fgr_ptr);
	Mat gmat(height, width, CV_32FC1, fgr_ptr + channel_step);
	Mat bmat(height, width, CV_32FC1, fgr_ptr + 2 * channel_step);
	Mat pmat(height, width, CV_32FC1, pha_ptr);
	rmat *= 255.;
	bmat *= 255.;
	gmat *= 255.;
	Mat rest = 1. - pmat;
	Mat mbmat = bmat.mul(pmat) + rest * 153.;
	Mat mgmat = gmat.mul(pmat) + rest * 255.;
	Mat mrmat = rmat.mul(pmat) + rest * 120.;
	std::vector<Mat> fgr_channel_mats, merge_channel_mats;
	fgr_channel_mats.push_back(bmat);
	fgr_channel_mats.push_back(gmat);
	fgr_channel_mats.push_back(rmat);
	merge_channel_mats.push_back(mbmat);
	merge_channel_mats.push_back(mgmat);
	merge_channel_mats.push_back(mrmat);

	content.pha_mat = pmat;
	merge(fgr_channel_mats, content.fgr_mat);
	merge(merge_channel_mats, content.merge_mat);
	content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
	content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

	content.flag = true;
}

void RobustVideoMatting::update_context(std::vector<Ort::Value>& output_tensors)
{
	// 0. update context for video matting.
	Ort::Value& r1o = output_tensors.at(2); // fgr (1,?,?h,?w)
	Ort::Value& r2o = output_tensors.at(3); // pha (1,?,?h,?w)
	Ort::Value& r3o = output_tensors.at(4); // pha (1,?,?h,?w)
	Ort::Value& r4o = output_tensors.at(5); // pha (1,?,?h,?w)
	auto r1o_dims = r1o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r2o_dims = r2o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r3o_dims = r3o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto r4o_dims = r4o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	// 1. update rxi's shape according to last rxo
	dynamic_input_node_dims.at(1) = r1o_dims;
	dynamic_input_node_dims.at(2) = r2o_dims;
	dynamic_input_node_dims.at(3) = r3o_dims;
	dynamic_input_node_dims.at(4) = r4o_dims;
	// 2. update rxi's value according to last rxo
	int64_t new_r1i_value_size = this->value_size_of(r1o_dims); // (1*?*?h*?w)
	int64_t new_r2i_value_size = this->value_size_of(r2o_dims); // (1*?*?h*?w)
	int64_t new_r3i_value_size = this->value_size_of(r3o_dims); // (1*?*?h*?w)
	int64_t new_r4i_value_size = this->value_size_of(r4o_dims); // (1*?*?h*?w)
	dynamic_r1i_value_handler.resize(new_r1i_value_size);
	dynamic_r2i_value_handler.resize(new_r2i_value_size);
	dynamic_r3i_value_handler.resize(new_r3i_value_size);
	dynamic_r4i_value_handler.resize(new_r4i_value_size);
	float* new_r1i_value_ptr = r1o.GetTensorMutableData<float>();
	float* new_r2i_value_ptr = r2o.GetTensorMutableData<float>();
	float* new_r3i_value_ptr = r3o.GetTensorMutableData<float>();
	float* new_r4i_value_ptr = r4o.GetTensorMutableData<float>();
	std::memcpy(dynamic_r1i_value_handler.data(), new_r1i_value_ptr, new_r1i_value_size * sizeof(float));
	std::memcpy(dynamic_r2i_value_handler.data(), new_r2i_value_ptr, new_r2i_value_size * sizeof(float));
	std::memcpy(dynamic_r3i_value_handler.data(), new_r3i_value_ptr, new_r3i_value_size * sizeof(float));
	std::memcpy(dynamic_r4i_value_handler.data(), new_r4i_value_ptr, new_r4i_value_size * sizeof(float));
	context_is_update = true;
}

void RobustVideoMatting::detect(Mat& mat, MattingContent& content, float downsample_ratio)
{
	if (mat.empty()) return;
	// 0. set dsr at runtime.
	dynamic_dsr_value_handler.at(0) = downsample_ratio;
	// 1. make input tensors, src, rxi, dsr
	std::vector<Ort::Value> input_tensors = this->transform(mat);
	// 2. inference, fgr, pha, rxo.
	auto output_tensors = session_->Run(
		Ort::RunOptions{ nullptr }, input_node_names.data(),
		input_tensors.data(), num_inputs, output_node_names.data(),
		num_outputs
	);
	// 3. generate matting
	this->generate_matting(output_tensors, content);
	// 4. update context (needed for video detection.)
	context_is_update = false; // init state.
	this->update_context(output_tensors);
}
