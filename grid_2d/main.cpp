#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdlib.h>

#include "kernel.h"

#define size_y 1000
#define size_x size_y*1.125

using namespace cv;
using namespace std;

int main(){

	Mat fractal_mat(size_y, size_x, CV_8U);

	// compute filter
	wrapper_gpu(fractal_mat);

	imwrite("output.jpg", fractal_mat);
	return 0;
}
