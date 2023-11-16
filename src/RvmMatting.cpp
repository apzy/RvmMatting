#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture("input.mp4");
	Mat frame;
	while (capture.read(frame))
	{
		imshow("frame", frame);
		waitKey(1);
	}
	return 0;
}

