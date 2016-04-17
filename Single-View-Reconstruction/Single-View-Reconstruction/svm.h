#include <iostream>  
#include <fstream>  
#include <stdio.h>
#include <opencv2/features2d/features2d.hpp>  
#include <opencv2/calib3d/calib3d.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

#include <vector>
#include "eigen.h"

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

cv::Mat myimage;

double cal3dZ(cv::Vec3d& v_x, cv::Vec3d& v_y, cv::Vec3d& v_z, cv::Vec3d& t, cv::Vec3d& b, cv::Vec3d& o, double scale)
{
	cv::Vec3d lxy = v_x.cross(v_y);
	double ret = o.dot(lxy) * cv::norm(b.cross(t)) / (b.dot(lxy) * cv::norm(v_z.cross(t))) * scale;
	return ret;
}

void cal3dXY(cv::Vec3d& b, cv::Mat& homo, double& x, double& y)
{
	Mat img_pt = Mat::ones(3, 1, CV_64F);
	img_pt.at<double>(0, 0) = b[0];
	img_pt.at<double>(1, 0) = b[1];
	Mat dst_pt = homo * img_pt;
	x = dst_pt.at<double>(0, 0) / dst_pt.at<double>(2, 0);
	y = dst_pt.at<double>(1, 0) / dst_pt.at<double>(2, 0);
}

cv::Vec3d cal3dXYZ(cv::Vec3d& v_x, cv::Vec3d& v_y, cv::Vec3d& v_z, cv::Vec3d& t, cv::Vec3d& b, cv::Vec3d& o, double scale, cv::Mat& homo)
{
	double x, y, z;
	z = cal3dZ(v_x, v_y, v_z, t, b, o, scale);
	cal3dXY(b, homo, x, y);
	return cv::Vec3d(x, y, z);
}


//inverse warping
cv::Mat getHomo(std::vector<cv::Vec2d>& src, std::vector<cv::Vec2d>& dst)
{
	std::vector<cv::Point2d> srcPoints, dstPoints;
	for (int i = 0; i < src.size(); i++)
	{
		cv::Point2d src_pnt(src[i][0], src[i][1]);
		cv::Point2d dst_pnt(dst[i][0], dst[i][1]);
		srcPoints.push_back(src_pnt);
		dstPoints.push_back(dst_pnt);
	}
	cv::Mat homo = cv::findHomography(srcPoints, dstPoints);
	return homo;
}

cv::Mat getHomo(std::vector<cv::Vec3d>& src, std::vector<cv::Vec3d>& dst)
{
	std::vector<cv::Point2d> srcPoints, dstPoints;
	for (int i = 0; i < src.size(); i++)
	{
		cv::Point2d src_pnt(src[i][0], src[i][1]);
		cv::Point2d dst_pnt(dst[i][0], dst[i][1]);
		srcPoints.push_back(src_pnt);
		dstPoints.push_back(dst_pnt);
	}
	cv::Mat homo = cv::findHomography(srcPoints, dstPoints);
	return homo;
}

void getTexture(cv::Mat& texture, cv::Mat& src, cv::Mat& homo)
{
	int width = texture.cols;
	int height = texture.rows;
	for (int w = 0; w < width; w++)
	{
		for (int h = 0; h < height; h++)
		{
			Mat dst_mat = Mat::ones(3, 1, CV_64F);
			dst_mat.at<double>(0, 0) = w;
			dst_mat.at<double>(1, 0) = h;
			Mat src_mat = homo * dst_mat;
			int src_w = int(src_mat.at<double>(0, 0) / src_mat.at<double>(2, 0));
			int src_h = int(src_mat.at<double>(1, 0) / src_mat.at<double>(2, 0));
			//std::cout << "dest: " << w << ", " << h << ", source: " << src_w << ", " << src_h << "\n";
			Vec3b color;
			if (src_w < src.cols && src_w >= 0 && src_h < src.rows && src_h >= 0)
			{
				color = src.at<Vec3b>(Point(src_w, src_h));
			}
			else
			{
				color = Vec3b(0, 0, 0);
			}
			texture.at<Vec3b>(Point(w, h)) = color;
		}
	}
}

static void onMouse(int event, int x, int y, int, void*){
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	std::cout << "click on (" << x << ", " << y << ")\n";
}

void add_point(float **m, CvPoint3D32f * p)
{
	m[0][0] += p->x * p->x;
	m[0][1] += p->x * p->y;
	m[0][2] += p->x * p->z;

	m[1][0] += p->x * p->y;
	m[1][1] += p->y * p->y;
	m[1][2] += p->y * p->z;

	m[2][0] += p->x * p->z;
	m[2][1] += p->y * p->z;
	m[2][2] += p->z * p->z;
}

CvPoint3D32f calc_vanishing_point(std::vector<CvPoint> points, float w)
{
	int num_lines = points.size() / 2;
	//compute lines
	CvPoint3D32f* lines = (CvPoint3D32f *)malloc(num_lines*sizeof(CvPoint3D32f));
	for (std::vector<int>::size_type i = 0; i != num_lines; ++i)
	{
		lines[i].x = points[2 * i].y / w - points[2 * i + 1].y / w;
		lines[i].y = points[2 * i + 1].x / w - points[2 * i].x / w;
		lines[i].z = (points[2 * i].x * points[2 * i + 1].y - points[2 * i + 1].x * points[2 * i].y) / (w * w);
	}


	//compute m
	float *eig_val = (float *)malloc(3 * sizeof(float));
	float **eig_vec = (float**)malloc(3 * sizeof(float*));
	float **m = (float**)malloc(3 * sizeof(float*));
	for (int i = 0; i<3; i++) {
		m[i] = (float *)malloc(3 * sizeof(float));
		eig_vec[i] = (float *)malloc(3 * sizeof(float));
	}
	for (int i = 0; i<3; i++) {
		for (int j = 0; j < 3; j++)
			m[i][j] = 0.0;
	}


	for (std::vector<int>::size_type i = 0; i != num_lines; ++i)
	{
		add_point(m, &lines[i]);
	}

	//compute Vertice
	eig_sys(3, m, eig_vec, eig_val);

	float min_eigen = 9999999.9;
	int index = -1;
	for (int i = 0; i < 3; i++)
		if (eig_val[i] < min_eigen)
		{
			min_eigen = eig_val[i];
			index = i;
		}
	float scale = w / eig_vec[index][2];
	return cvPoint3D32f(eig_vec[index][0] * scale, eig_vec[index][1] * scale, eig_vec[index][2] * scale);
}

double distance2d(cv::Vec3d v1, cv::Vec3d v2)
{
	return std::sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2));
}

double distance3d(cv::Vec3d v1, cv::Vec3d v2)
{
	return std::sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2) + pow(v1[2] - v2[2], 2));
}

int create_crml_file(std::vector<cv::Vec3d>& points, std::ofstream &out, std::string s[]){

	if (out.is_open())
	{
		for (unsigned int i = 0; i < points.size(); i++){
			if (i % 4 == 0){
				out << "#VRML V1.0 ascii" << std::endl;
				out << "#TargetJr VRML_IO output" << std::endl;
				out << "Separator { \nShapeHints { \n\tvertexOrdering  CLOCKWISE \n\tshapeType \tSOLID \n}" << std::endl;
				out << "Separator {\n\tCoordinate3 { point [" << std::endl;
			}

			out << "\t\t" << points[i][0] << "\t" << points[i][1] << "\t" << points[i][2] << "\n";

			if (i % 4 == 3){
				out << "\t]}" << std::endl;
				out << "\tTexture2 { filename " << " \" " << s[i / 4] << " \" }" << "\n";
				out << "\tTextureCoordinate2 { point [" << "\n";
				out << "\t5.87859e-16 0," << "\n";
				out << "\t0 1," << "\n";
				out << "\t1 1," << "\n";
				out << "\t1 2.22045e-15," << "\n";
				out << "\t]}" << "\n";
				out << "\tIndexedFaceSet { coordIndex [" << "\n";
				out << "\t0, 1, 2, 3, -1," << "\n";
				out << "\t]}" << "\n";
				out << "}" << "\n";
				out << "}" << "\n";

				out << "# End TargetJr VRML_IO output" << "\n\n";
			}
		}
		return 0;
	}
	else
		return -1;
}