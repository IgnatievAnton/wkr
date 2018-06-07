#include <opencv\highgui.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include<opencv\cv.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <vector>
#include<fstream>
#pragma comment(lib, "../../../../opencv/build/x64/vc14/lib/opencv_world310.lib")
#pragma comment(lib, "../../../../opencv/build/x64/vc14/lib/opencv_world310d.lib")
using namespace std;
using namespace cv;

void mat2uchar(Mat mat_image, unsigned char*& byte_image, int& width, int& height)
{

	width = mat_image.size().width;
	height = mat_image.size().height;
	byte_image = (unsigned char*)calloc(sizeof(char) * width * height, 1);
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			uchar& uxy = mat_image.at<uchar>(j, i);
			int color = (int16_t)uxy;
			byte_image[j * width + i] = color;
		}
	}
}

void uchar2mat(unsigned char* byte_image, int& width, int& height, Mat& mat_image)
{
	mat_image = Mat::zeros(height, width, CV_8U);
	width = mat_image.size().width;
	height = mat_image.size().height;
	for (int j = 0; j < height; j++)
	{	
		for (int i = 0; i < width; i++)
		{
			mat_image.at<uchar>(j, i) = byte_image[j * width + i];
		}
	}
}


void NDWI(string nameImg1,string nameImg2)
{
	fstream file("ndwi.txt", ios::out);
	int sizzzz = 0;
	Mat image = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image2 = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image3 = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imah;
	Mat imah2;
	double c = 0.0;
	double first = 0.0;
	double second = 0.0;
	double v = 0.0;
	unsigned char *ima;
	const unsigned int sizeH = image.rows;
	double **value = new  double*[sizeH];
	for (int i = 0; i < image2.rows; i++)
	{
		value[i] = new  double[sizeH];
	}
	int w = 0;
	int h = 0;
	mat2uchar(image, ima, w, h);
	uchar2mat(ima, w, h, imah);

	mat2uchar(image2, ima, w, h);
	uchar2mat(ima, w, h, imah2);

	const unsigned int sizeh = image2.rows;
	const unsigned int sizew = image2.cols;
	double firstTest = 0.0;
	double secondTest = 0.0;

	cout << "SizeH: " << sizeh << endl;
	cout << "SizeW: " << sizew << endl;


	double NDWI = 0.0;
	for (int i = 0; i <sizeh; i++)
	{
		for (int k = 0; k < sizew; k++)
		{
			int green = 0;
			int nir = 0;
			uchar&uxy = imah.at<uchar>(i, k);
			green = (int)uxy;
			uchar&xy = imah2.at<uchar>(i, k);
			nir = (int)xy;

			firstTest = green - nir;
			secondTest = green + nir;
			if (secondTest > 0)
			{
				NDWI = firstTest / secondTest;
				value[i][k] = (double)NDWI;
				w++;
				if (value[i][k] != 0 && value[i][k] != 1 && value[i][k] != -1)
				{

				}
			}
			else
			{
				h++;
				value[i][k] = 0;
			}

		}
	}

	for (int i = 0; i < sizeh; i++)
	{

		for (int k = 0; k <sizew; k++)
		{
			file << value[i][k];
			if (value[i][k] > 0 && value[i][k] <1)
			{
				uchar abc = 255;
				image3.at<uchar>(i, k) = abc;
			}
			else
			{
				uchar abc = 0;
				image3.at<uchar>(i, k) = abc;
			}
		}
	}

	imwrite("NDWI_karlsruhe_2003.bmp", image3);
	cout << "NDWI - done! " << endl;
}


void NDVI(string nameImg1, string nameImg2)
{
	int sizzzz = 0;
	Mat image = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image2 = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image3 = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imah;
	Mat imah2;
	double c = 0.0;
	double first = 0.0;
	double second = 0.0;
	double v = 0.0;
	unsigned char *ima;
	fstream file("1.txt", ios::out);
	const unsigned int sizeH = image.rows;
	double **value = new  double*[sizeH];
	for (int i = 0; i < image2.rows; i++)
	{
		value[i] = new  double[sizeH];
	}
	int w = 0;
	int h = 0;
	mat2uchar(image, ima, w, h);
	uchar2mat(ima, w, h, imah);

	mat2uchar(image2, ima, w, h);
	uchar2mat(ima, w, h, imah2);

	const unsigned int sizeh = image2.rows;
	const unsigned int sizew = image2.cols;
	double firstTest = 0.0;
	double secondTest = 0.0;

	cout << "SizeH: " << sizeh << endl;
	cout << "SizeW: " << sizew << endl;


	double NDVI = 0.0;
	for (int i = 0; i <sizeh; i++)
	{
		for (int k = 0; k < sizew; k++)
		{
			int nir = 0;
			int red = 0;
			uchar&uxy = imah.at<uchar>(i, k);
			nir = (int)uxy;
			uchar&xy = imah2.at<uchar>(i, k);
			red = (int)xy;

			firstTest = nir - red;
			secondTest = nir + red;
			if (secondTest > 0)
			{
				NDVI = firstTest / secondTest;
				value[i][k] = (double)NDVI;
				w++;
				if (value[i][k] != 0 && value[i][k] != 1 && value[i][k] != -1)
				{

				}
			}
			else
			{
				h++;
				value[i][k] = 0;
			}

		}
	}


	for (int i = 0; i < sizeh; i++)
	{

		for (int k = 0; k <sizew; k++)
		{

			if (value[i][k] >0.4 && value[i][k]<1)
			{
				uchar abc = 255;
				image3.at<uchar>(i, k) = abc;
			}
			if (value[i][k] > 0 && value[i][k] < 0.4)
			{
				uchar abc = 144;
				image3.at<uchar>(i, k) = abc;
			}
			if (value[i][k] > -1 && value[i][k] == 0)
			{
				uchar abc = 0;
				image3.at<uchar>(i, k) = abc;

			}
		}
	}

	imwrite("NDVI_karlsruhe_2003.bmp", image3);
	cout << "NDVI - done! " << endl;
}


void NDVI_NDWI(string nameImg1, string nameImg2)
{
	Mat NDVI = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat NDWI = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < NDVI.rows; i++)
	{
		for (int k = 0; k < NDVI.cols; k++)
		{
			int ndvi = NDVI.at<uchar>(i, k);
			int ndwi = NDWI.at<uchar>(i, k);
			if (ndvi==144 || ndvi == 0)
			{
				if (ndwi == 255)
				{
					NDVI.at<uchar>(i, k) = 144;
				}
			}
		}
	}
	imwrite("NDVI+NDWI_karlsruhe_2003.bmp", NDVI);
	cout << "NDVI+NDWI DONE! " << endl;
}


void ARVI(string nameImg1, string nameImg2, string nameImg3)
{
	int sizzzz = 0;
	Mat image = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image2 = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image3 = imread(nameImg3, CV_LOAD_IMAGE_GRAYSCALE);
	Mat image4 = imread(nameImg3, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imah;
	Mat imah2;
	Mat imah3;
	double c = 0.0;
	double first = 0.0;
	double second = 0.0;
	double v = 0.0;
	unsigned char *ima;
	fstream file("1.txt", ios::out);
	const unsigned int sizeH = image.rows;
	double **value = new  double*[sizeH];
	for (int i = 0; i < image2.rows; i++)
	{
		value[i] = new  double[sizeH];
	}
	int w = 0;
	int h = 0;
	mat2uchar(image, ima, w, h);
	uchar2mat(ima, w, h, imah);

	mat2uchar(image2, ima, w, h);
	uchar2mat(ima, w, h, imah2);

	mat2uchar(image3, ima, w, h);
	uchar2mat(ima, w, h, imah3);

	const unsigned int sizeh = image2.rows;
	const unsigned int sizew = image2.cols;
	double firstTest = 0.0;
	double secondTest = 0.0;

	double ARVI = 0.0;
	for (int i = 0; i <sizeh; i++)
	{
		for (int k = 0; k < sizew; k++)
		{
			int nir = 0;
			int red = 0;
			int blue = 0;
			uchar&uxy = imah.at<uchar>(i, k);
			nir = (int)uxy;
			uchar&xy = imah2.at<uchar>(i, k);
			red = (int)xy;
			uchar&xyu = imah3.at<uchar>(i, k);
			blue = (int)xyu;
			double Rb = red - 1 * (red - blue);
			firstTest = nir - Rb;
			secondTest = nir + Rb;
			if (secondTest > 0)
			{
				ARVI = firstTest / secondTest;
				value[i][k] = (double)ARVI;
				w++;
				if (value[i][k] != 0 && value[i][k] != 1 && value[i][k] != -1)
				{

				}
			}
			else
			{
				h++;
				value[i][k] = 0;
			}

		}
	}


	for (int i = 0; i < sizeh; i++)
	{

		for (int k = 0; k <sizew; k++)
		{

			if (value[i][k] > 0.3)
			{
				uchar abc = 255;
				image4.at<uchar>(i, k) = abc;
			}
			else
				image4.at<uchar>(i, k) = 0;
		}
	}

	imwrite("ARVI_karlsruhe_2003.bmp", image4);
	cout << "ARVI - done! " << endl;
}


void CheckNDVI(string nameImg1)
{
	Mat imageFirst = imread("new2001.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	const unsigned int sizehFirst = imageFirst.rows;
	const unsigned int sizewFirst = imageFirst.cols;
	Mat imageSecond = imread("new2002.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	const unsigned int sizehSecond = imageSecond.rows;
	const unsigned int sizewSecond = imageSecond.cols;
	Mat image3(imageFirst.rows, imageFirst.cols, CV_8UC1);
	image3 = imageFirst;
	cvtColor(imageFirst, image3, CV_GRAY2RGB);
	if (sizehFirst != sizehSecond && sizewFirst != sizewSecond)
	{
	cout << "Error:Images have diferent size1231231!";
	return;
	}
	Mat imah;
	Mat imah2;
	double c = 0.0;
	double first = 0.0;
	double second = 0.0;
	double v = 0.0;
	unsigned char *ima;
	double **value1 = new  double*[sizehFirst];
	for (int i = 0; i < sizehSecond; i++)
	{
	value1[i] = new  double[sizehSecond];
	}
	double **value2 = new  double*[sizehFirst];
	for (int i = 0; i < sizehSecond; i++)
	{
	value2[i] = new  double[sizehSecond];
	}
	int w = 0;
	int h = 0;
	mat2uchar(imageFirst, ima, w, h);
	uchar2mat(ima, w, h, imah);
	mat2uchar(imageSecond, ima, w, h);
	uchar2mat(ima, w, h, imah2);


	for (int i = 0; i < sizehFirst; i++)
	{
		for (int k = 0; k < sizewFirst; k++)
		{
			int first = 0;
			int second = 0;
			uchar&uxy = imah.at<uchar>(i, k);
			first = (int)uxy;
			uchar&xy = imah2.at<uchar>(i, k);
			second = (int)xy;
			value1[i][k] = (double)first;
			value2[i][k] = (double)second;
		}
	}
	for (int i = 0; i < sizehFirst; i++)
	{

	for (int k = 0; k < sizewFirst; k++)
	{
		if (value1[i][k] > value2[i][k]+30 || value1[i][k] < value2[i][k] -30)
			{
					image3.at<Vec3b>(i, k) = Vec3b(29, 204, 55);
			}
	}
	}
	imwrite("dzz1.bmp", image3);
}

void NDVI_NDWI_ARV(string nameImg1, string nameImg2)
{
	Mat NDVI_NDWI = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat ARVI = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < NDVI_NDWI.rows; i++)
	{
		for (int k = 0; k < NDVI_NDWI.cols; k++)
		{
			int ndvi = NDVI_NDWI.at<uchar>(i, k);
			int arvi = ARVI.at<uchar>(i, k);
			if (ndvi == 144 || ndvi == 0)

			{
				if (arvi == 0)
				{
					NDVI_NDWI.at<uchar>(i, k) = 0;
				}
			}
		}
	}
	imwrite("NDVI+NDWI+Arvi_karlsruhe_2003.bmp", NDVI_NDWI);
	cout << "NDVI+NDWI+ARVI DONE! " << endl;
}


void ImageCreate(string nameImg1, string nameImg2, string nameImg3)
{
	Mat r = imread(nameImg1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat g = imread(nameImg2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat b = imread(nameImg3, CV_LOAD_IMAGE_GRAYSCALE);
	Mat rgb (b.size(), CV_8UC3);
	rgb = 0;
	uchar* start = rgb.data;
	for (int y = 0; y < r.rows; y++)
	{
		for (int x = 0; x < r.cols; x++)
		{
			*(start++) = r.at<uchar>(y, x);
			*(start++) = g.at<uchar>(y, x);
			*(start++) = b.at<uchar>(y, x);
		}
	}
	rgb *= 1.5;
	imwrite("check.bmp", rgb);
}


void obrez()
{
	Mat r = imread("NDVI+NDWI+Arvi_karlsruhe_2001.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat g = imread("NDVI+NDWI+Arvi_karlsruhe_2003.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat b(g.rows-2,r.cols-2, CV_8UC1);
	b = 0;
	Mat d(g.rows - 2, r.cols - 2, CV_8UC1);
	d = 0;
	for (int i = 0; i < b.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			b.at<uchar>(i, j) = g.at<uchar>(i + 2, j + 2);
		}
	}
	for (int i = 0; i < b.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			d.at<uchar>(i, j) = r.at<uchar>(i , j );
		}
	}
	imwrite("new2001.bmp", b);
	imwrite("new2002.bmp", d);
}

int main(int argc, char* argv[])
{
	string Nir = "2001_nir.bmp";
	string Red = "2001_red.bmp";
	string Green = "2003_grn.bmp";
	string blue = "2003_blue.bmp";
	ImageCreate(Red, Green, blue);

	NDVI(Nir,Red);
	NDWI(Green, Nir);
	string NDVI = "NDVI_karlsruhe_2003.bmp";
	string NDWI = "NDWI_karlsruhe_2003.bmp";
	string ndvi_ndwi= "NDVI+NDWI_karlsruhe_2003.bmp";
	string arvi = "ARVI_karlsruhe_2003.bmp";
	CheckNDVI(NDVI);
	NDVI_NDWI(NDVI, NDWI);
	ARVI(Nir, Red, blue);
	NDVI_NDWI_ARV(ndvi_ndwi,arvi);
	obrez();
	std::cin.get();
	return 0;
}
