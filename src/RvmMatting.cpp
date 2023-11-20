#include <iostream>
#include <opencv2/opencv.hpp>
#include "RVM.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	string inputPath = argv[1];
	inputPath = inputPath.substr(0, inputPath.length() - 4);

	printf("input file name => %s\n", inputPath.c_str());
	VideoCapture capture(inputPath + ".mp4");
	int fps = capture.get(CAP_PROP_FPS);
	printf("fps = %d\n", fps);
	VideoWriter videoWriter = cv::VideoWriter(inputPath + "_out.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(1080, 1920));
	Mat frame, result;
	RVM rvm;
	while (capture.read(frame))
	{
		rvm.inference(frame, result);
		videoWriter.write(result);
	}
	videoWriter.release();
	return 0;
}

