#include <iostream>  
#include <stdio.h>
#include <opencv2/features2d/features2d.hpp>  
#include <opencv2/calib3d/calib3d.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

using namespace std;
using namespace cv;

cv::Mat myimage;

double cal3dZ(cv::Vec3d& v_x, cv::Vec3d& v_y, cv::Vec3d& v_z, cv::Vec3d& t, cv::Vec3d& b, cv::Vec3d& o, double scale)
{
	cv::Vec3d lxy = v_x.cross(v_y);
	double ret = o.dot(lxy) * cv::norm(b.cross(t)) / (b.dot(lxy) * cv::norm(v_z.cross(t))) * scale;
	return ret;
}

void cal3dXY(cv::Vec3d& b, cv::Mat homo, double& x, double& y)
{
	Mat img_pt = Mat::ones(3, 1, CV_64F);
	img_pt.at<double>(0, 0) = b[0];
	img_pt.at<double>(1, 0) = b[1];
	Mat dst_pt = homo * img_pt;
	x = dst_pt.at<double>(0, 0) / dst_pt.at<double>(2, 0);
	y = dst_pt.at<double>(1, 0) / dst_pt.at<double>(2, 0);
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
			std::cout << "dest: " << w << ", " << h << ", source: " << src_w << ", " << src_h << "\n";
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

	cv::Point center(x, y);
	cv::circle(myimage, center, 2, cv::Scalar(0, 0, 255), CV_FILLED);
	cv::imshow("src", myimage);

	std::cout << "click on (" << x << ", " << y << ")\n";
}