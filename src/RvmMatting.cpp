#include <iostream>
#include <opencv2/opencv.hpp>
#include  "RobustVideoMatting.h"

using namespace std;
using namespace cv;

int main()
{
	const float downsample_ratio = 0.4;
	string model_path = "rvm_mobilenetv3_fp32.onnx";
	RobustVideoMatting rvm(model_path);
	VideoCapture capture("input.mp4");
	Mat frame;
	while (capture.read(frame))
	{
		MattingContent content;
		rvm.detect(frame, content, downsample_ratio);
		imshow("frame", content.merge_mat);
		waitKey(1);
	}
	return 0;
}

