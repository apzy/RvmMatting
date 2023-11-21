#include "RobustVideoMatting.h"
#include <chrono>
#include <omp.h>

inline float float16_to_float32(uint16_t v)
{
	union
	{
		uint32_t u; float f;
	} t;

	t.u = v;
	t.u = ((t.u & 0x7fff) << 13) + 0x38000000;
	t.u |= ((v & 0x8000) << 16);
	return t.f;
}

inline uint16_t float32_to_float16(float v)
{
	union
	{
		uint32_t u; float f;
	} t;
	uint16_t y;

	t.f = v;
	y = ((t.u & 0x7fffffff) >> 13) - (0x38000000 >> 13);
	y |= ((t.u & 0x80000000) >> 16);
	return y;
}

RobustVideoMatting::RobustVideoMatting(string model_path)
{
	wstring widestr = wstring(model_path.begin(), model_path.end());
	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
	session_ = new Session(env, widestr.c_str(), sessionOptions);
}

void RobustVideoMatting::normalize_(Mat img, vector<uint16_t>& output)
{
	int row = img.rows;
	int col = img.cols;

#pragma omp parallel for collapse(2)  
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float pix = img.ptr<uchar>(i)[j * 3 + 2 - 0];
			output[0 * row * col + i * col + j] = float32_to_float16(pix / 255.0);
			pix = img.ptr<uchar>(i)[j * 3 + 2 - 1];
			output[1 * row * col + i * col + j] = float32_to_float16(pix / 255.0);
			pix = img.ptr<uchar>(i)[j * 3 + 2 - 2];
			output[2 * row * col + i * col + j] = float32_to_float16(pix / 255.0);
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
	auto beginTime = std::chrono::high_resolution_clock::now();
	Mat src = mat.clone();
	const unsigned int img_height = mat.rows;
	const unsigned int img_width = mat.cols;
	vector<int64_t>& src_dims = dynamic_input_node_dims.at(0); // (1,3,h,w)
	src_dims.at(2) = img_height;
	src_dims.at(3) = img_width;
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);

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

	input_tensors.push_back(Value::CreateTensor(allocator_info, dynamic_src_value_handler.data(), dynamic_src_value_handler.size() * sizeof(uint16_t), src_dims.data(), src_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
	input_tensors.push_back(Value::CreateTensor(allocator_info, dynamic_r1i_value_handler.data(), r1i_value_size * sizeof(uint16_t), r1i_dims.data(), r1i_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
	input_tensors.push_back(Value::CreateTensor(allocator_info, dynamic_r2i_value_handler.data(), r2i_value_size * sizeof(uint16_t), r2i_dims.data(), r2i_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
	input_tensors.push_back(Value::CreateTensor(allocator_info, dynamic_r3i_value_handler.data(), r3i_value_size * sizeof(uint16_t), r3i_dims.data(), r3i_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
	input_tensors.push_back(Value::CreateTensor(allocator_info, dynamic_r4i_value_handler.data(), r4i_value_size * sizeof(uint16_t), r4i_dims.data(), r4i_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
	input_tensors.push_back(Value::CreateTensor<float>(allocator_info, dynamic_dsr_value_handler.data(), dsr_value_size, dsr_dims.data(), dsr_dims.size()));

	return input_tensors;
}

void RobustVideoMatting::generate_matting(std::vector<Ort::Value>& output_tensors, MattingContent& content)
{
	Ort::Value& pha = output_tensors.at(1);
	auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	const unsigned int height = pha_dims.at(2);
	const unsigned int width = pha_dims.at(3);
	const unsigned int channel_step = height * width;
	uint16_t* pha_ptr = pha.GetTensorMutableData<uint16_t>();
	int r = 0;
	int g = 255;
	int b = 0;
	content.merge_mat = cv::Mat(height, width, CV_8UC3, cv::Scalar(r, g, b));

	int step = width * 3;
#pragma omp parallel for collapse(2)
	for (int j = 0; j < height; ++j)
	{
		for (int i = 0, k = 0; i < step; i += 3,++k)
		{
			float data = float16_to_float32(*(pha_ptr + j * width + k));

			if (data >= 1 )
			{
				content.merge_mat.ptr<uchar>(j)[i + 0] = content.fgr_mat.ptr<uchar>(j)[i + 0];
				content.merge_mat.ptr<uchar>(j)[i + 1] = content.fgr_mat.ptr<uchar>(j)[i + 1];
				content.merge_mat.ptr<uchar>(j)[i + 2] = content.fgr_mat.ptr<uchar>(j)[i + 2];
			}
			else if (data > 0)
			{
				content.merge_mat.ptr<uchar>(j)[i + 0] = data * content.fgr_mat.ptr<uchar>(j)[i + 0] + (1 - data) * content.merge_mat.ptr<uchar>(j)[i + 0];
				content.merge_mat.ptr<uchar>(j)[i + 1] = data * content.fgr_mat.ptr<uchar>(j)[i + 1] + (1 - data) * content.merge_mat.ptr<uchar>(j)[i + 1];
				content.merge_mat.ptr<uchar>(j)[i + 2] = data * content.fgr_mat.ptr<uchar>(j)[i + 2] + (1 - data) * content.merge_mat.ptr<uchar>(j)[i + 2];
			}
		}
	}
	//Mat rmat(height, width, CV_32FC1, fgr_ptr);
	//Mat gmat(height, width, CV_32FC1, fgr_ptr + channel_step);
	//Mat bmat(height, width, CV_32FC1, fgr_ptr + 2 * channel_step);
	//rmat *= 255.;
	//bmat *= 255.;
	//gmat *= 255.;
	//Mat rest = 1 - pmat;
	//Mat mbmat = bmat.mul(pmat) + rest * 153.;
	//Mat mgmat = gmat.mul(pmat) + rest * 255.;
	//Mat mrmat = rmat.mul(pmat) + rest * 120.;
	//std::vector<Mat> fgr_channel_mats, merge_channel_mats;
	//fgr_channel_mats.push_back(bmat);
	//fgr_channel_mats.push_back(gmat);
	//fgr_channel_mats.push_back(rmat);
	//merge_channel_mats.push_back(mbmat);
	//merge_channel_mats.push_back(mgmat);
	//merge_channel_mats.push_back(mrmat);

	//content.pha_mat = pmat;
	//merge(fgr_channel_mats, content.fgr_mat);
	//merge(merge_channel_mats, content.merge_mat);
	//content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
	//content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

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
	uint16_t* new_r1i_value_ptr = r1o.GetTensorMutableData<uint16_t>();
	uint16_t* new_r2i_value_ptr = r2o.GetTensorMutableData<uint16_t>();
	uint16_t* new_r3i_value_ptr = r3o.GetTensorMutableData<uint16_t>();
	uint16_t* new_r4i_value_ptr = r4o.GetTensorMutableData<uint16_t>();
	std::memcpy(dynamic_r1i_value_handler.data(), new_r1i_value_ptr, new_r1i_value_size * sizeof(uint16_t));
	std::memcpy(dynamic_r2i_value_handler.data(), new_r2i_value_ptr, new_r2i_value_size * sizeof(uint16_t));
	std::memcpy(dynamic_r3i_value_handler.data(), new_r3i_value_ptr, new_r3i_value_size * sizeof(uint16_t));
	std::memcpy(dynamic_r4i_value_handler.data(), new_r4i_value_ptr, new_r4i_value_size * sizeof(uint16_t));
	context_is_update = true;
}

void RobustVideoMatting::detect(Mat& mat, MattingContent& content, float downsample_ratio)
{
	if (mat.empty()) return;
	content.fgr_mat = mat;
	// 0. set dsr at runtime.
	auto beginTime = std::chrono::high_resolution_clock::now();

	dynamic_dsr_value_handler.at(0) = downsample_ratio;
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	beginTime = std::chrono::high_resolution_clock::now();

	// 1. make input tensors, src, rxi, dsr
	std::vector<Ort::Value> input_tensors = this->transform(mat);
	endTime = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	printf("transform time used = %d\n", elapsedTime.count());
	beginTime = std::chrono::high_resolution_clock::now();

	// 2. inference, fgr, pha, rxo.
	auto outputTensor = session_->Run(
		Ort::RunOptions{ nullptr }, input_node_names.data(),
		input_tensors.data(), num_inputs, output_node_names.data(),
		num_outputs
	);
	endTime = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	printf("run time used = %d\n", elapsedTime.count());
	beginTime = std::chrono::high_resolution_clock::now();

	// 3. generate matting
	this->generate_matting(outputTensor, content);
	endTime = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	printf("generate time used = %d\n", elapsedTime.count());
	beginTime = std::chrono::high_resolution_clock::now();

	// 4. update context (needed for video detection.)
	context_is_update = false; // init state.
	this->update_context(outputTensor);
	endTime = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	printf("context time used = %d\n", elapsedTime.count());
	printf("----------------------------------------------------------------------------------\n");
}
