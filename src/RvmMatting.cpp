#include <iostream>
#include <opencv2/opencv.hpp>
#include "RVM.h"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture("input.mp4");
	Mat frame,result;
	RVM rvm;
	while (capture.read(frame))
	{
		rvm.inference(frame, result);
		imshow("frame", frame);
		waitKey(1);
		imshow("result", result);
		waitKey(1);
	}
	return 0;
}

