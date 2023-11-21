#include <iostream>
#include <opencv2/opencv.hpp>
#include  "RobustVideoMatting.h"

using namespace std;
using namespace cv;

int main()
{
	const float downsample_ratio = 0.4;
	string model_path = "rvm_resnet50_fp16.onnx";
	RobustVideoMatting rvm(model_path);
	VideoCapture capture("input.mp4");
	Mat frame;
	while (capture.read(frame))
	{
		MattingContent content;
	auto beginTime = std::chrono::high_resolution_clock::now();
		rvm.detect(frame, content, downsample_ratio);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime);
	printf("============================time used = %d\n", elapsedTime.count());
		imshow("frame", content.merge_mat);
		waitKey(1);
	}
	return 0;
}

