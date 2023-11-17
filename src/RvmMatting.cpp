#include <iostream>
#include <opencv2/opencv.hpp>
#include "RVM.h"

using namespace std;
using namespace cv;

inline void audio_extraction(string path)
{
	string cmd = "ffmpeg -i " + path + ".mp4 -f wav -ar 16000 " + path + ".wav -y";
	printf("extract cmd = %s\n", cmd.c_str());
	system(cmd.c_str());
}

inline void audio_synthesis(string path)
{
	string cmd = "ffmpeg -i " + path + "_out.mp4 -i " + path + ".wav  -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 " + path + "_audio.mp4 -y";
	printf("extract cmd = %s\n", cmd.c_str());
	system(cmd.c_str());
}


int main(int argc, char* argv[])
{
	string inputPath = argv[1];
	printf("input file name => %s\n", inputPath.c_str());
	audio_extraction(inputPath);
	VideoCapture capture(inputPath + ".mp4");
	int fps = capture.get(CAP_PROP_FPS);
	printf("fps = %d\n", fps);
	VideoWriter videoWriter = cv::VideoWriter(inputPath + "_out.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(1080,1920));
	Mat frame, result;
	RVM rvm;
	while (capture.read(frame))
	{
		rvm.inference(frame, result);
		videoWriter.write(result);
	}
	videoWriter.release();
	audio_synthesis(inputPath);
	return 0;
}

