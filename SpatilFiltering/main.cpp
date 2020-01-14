#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\imgproc.hpp>
#include<iostream>
#include <string>

using namespace cv;
using namespace std;

class SpatialFiltering {
private:
	Mat padding(Mat img, int k_width, int k_height, string type)
	{
		Mat scr;
		img.convertTo(scr, CV_64FC1);
		int pad_rows, pad_cols;
		pad_rows = (k_height - 1) / 2;
		pad_cols = (k_width - 1) / 2;
		Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
		scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows)));
		if (type == "mirror")
		{
			for (int i = 0; i < pad_rows; i++)
			{
				scr(Rect(0, pad_rows - i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				scr(Rect(0, (scr.rows - 1) - pad_rows + i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, (pad_image.rows - 1) - i, scr.cols, 1)));
			}

			for (int j = 0; j < pad_cols; j++)
			{
				pad_image(Rect(2 * pad_cols - j, 0, 1, pad_image.rows)).copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
				pad_image(Rect((pad_image.cols - 1) - 2 * pad_cols + j, 0, 1, pad_image.rows)).copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			}

			return pad_image;
		}
		else if (type == "replicate")
		{
			for (int i = 0; i < pad_rows; i++)
			{
				scr(Rect(0, 0, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				scr(Rect(0, (scr.rows - 1), scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, (pad_image.rows - 1) - i, scr.cols, 1)));
			}

			for (int j = 0; j < pad_cols; j++)
			{
				pad_image(Rect(pad_cols, 0, 1, pad_image.rows)).copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
				pad_image(Rect((pad_image.cols - 1) - pad_cols, 0, 1, pad_image.rows)).copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			}

			return pad_image;
		}
		else
		{
			return pad_image;
		}
	}

	Mat define_kernel(int k_width, int k_height, string type) {

		if (type == "box")
		{
			//Mat (int rows, int cols, int type)
			Mat kernel(k_height, k_width, CV_64FC1, Scalar(1.0 / (k_width * k_height)));
			return kernel;
		}
		else if (type == "guassian")
		{
			int pad_rows = (k_height - 1) / 2;
			int pad_cols = (k_width - 1) / 2;
			Mat kernel(k_height, k_width, CV_64FC1);
			for (int i = -pad_rows; i <= pad_rows; i++)
			{
				for (int j = -pad_cols; j <= pad_cols; j++)
				{
					kernel.at <double>(i + pad_rows, j + pad_cols) = exp(-(i*i + j*j) / 2.0);
				}
			}
			kernel = kernel / sum(kernel);
			return kernel;
		}

	}

public:
	void convolve(Mat scr, Mat &dst, int k_w, int k_h, string padingType, string filterType)
	{
		Mat pad_img, kernel;
		pad_img = padding(scr, k_w, k_h, padingType);
		kernel = define_kernel(k_w, k_h, filterType);

		Mat output = Mat::zeros(scr.size(), CV_64FC1);
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, k_w, k_h)))).val[0];
			}
		}

		output.convertTo(dst, CV_8UC1);
	}
};


int main() {
	Mat img, dst;
	img = imread("airship.jpg", 0);


	Mat kernel;
	double k_w = 3;
	double k_h = 3;


	SpatialFiltering F1;
	F1.convolve(img, dst, k_w, k_h, "mirror", "guassian");

	Mat built;
	GaussianBlur(img, built, Size(3, 3), 1, 1);

	namedWindow("built", WINDOW_AUTOSIZE);
	imshow("built", built);

	namedWindow("dst", WINDOW_AUTOSIZE);
	imshow("dst", dst);

	//imwrite("out.png", dst);
	//imwrite("built.png", built);

	waitKey(0);
	return 0;
}


/*
Mat img, output;
// read image file
img = imread("airship.jpg", 0);
blur(img, output, Size(50, 50));

namedWindow("output", WINDOW_AUTOSIZE);
imshow("output", output);


waitKey(0);
return 0;
*/
