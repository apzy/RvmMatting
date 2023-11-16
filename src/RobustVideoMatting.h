#pragma once
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Ort;

typedef struct MattingContentType
{
	Mat fgr_mat; // fore ground mat 3 channel (R,G,B) 0.~1. or 0~255
	Mat pha_mat; // alpha(matte) 0.~1.
	Mat merge_mat; // merge bg and fg according pha
	bool flag;

	MattingContentType() : flag(false)
	{};
} MattingContent;

class RobustVideoMatting
{
public:
	RobustVideoMatting(string model_path);
	void detect(Mat& mat, MattingContent& content, float downsample_ratio);
private:
	Session* session_;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "robustvideomatting");
	SessionOptions sessionOptions = SessionOptions();

	unsigned int num_inputs = 6;
	vector<const char*> input_node_names = {
		"src",
		"r1i",
		"r2i",
		"r3i",
		"r4i",
		"downsample_ratio"
	};
	// init dynamic input dims
	vector<vector<int64_t>> dynamic_input_node_dims = {
		{1, 3, 1280, 720}, // src  (b=1,c,h,w)
		{1, 1, 1,    1}, // r1i
		{1, 1, 1,    1}, // r2i
		{1, 1, 1,    1}, // r3i
		{1, 1, 1,    1}, // r4i
		{1} // downsample_ratio dsr
	}; // (1, 16, ?h, ?w) for inner loop rxi

	// hardcode output node names
	unsigned int num_outputs = 6;
	vector<const char*> output_node_names = {
		"fgr",
		"pha",
		"r1o",
		"r2o",
		"r3o",
		"r4o"
	};
	// input values handler & init
	vector<float> dynamic_src_value_handler;
	vector<float> dynamic_r1i_value_handler = { 0.0f }; // init 0. with shape (1,1,1,1)
	vector<float> dynamic_r2i_value_handler = { 0.0f };
	vector<float> dynamic_r3i_value_handler = { 0.0f };
	vector<float> dynamic_r4i_value_handler = { 0.0f };
	vector<float> dynamic_dsr_value_handler = { 0.25f }; // downsample_ratio with shape (1)
	int64_t value_size_of(const std::vector<int64_t>& dims);
	bool context_is_update = false;
	void normalize_(Mat img, vector<float>& output);
	vector<Ort::Value> transform(const Mat& mat);
	void generate_matting(vector<Ort::Value>& output_tensors, MattingContent& content);
	void update_context(vector<Ort::Value>& output_tensors);
};
