#include "svm.h"

using namespace std;
using namespace cv;

int main()
{
	std::string image_path = "1.jpg";
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		std::cout << "Fail to open image\n";
		return -1;
	}
	std::cout << "src image: cols = " << image.cols << "rows = " << image.rows <<"\n";

/*	cv::namedWindow("Painting", WINDOW_NORMAL);
	cv::setMouseCallback("Painting", onMouse, 0);
	cv::imshow("Painting", image);
	cv::waitKey(0);
	return 0 ;
*/

	Vec2d p1(539, 809), p2(654, 667), p3(815, 666), p4(842, 807);
	int width = (int)(sqrt(pow(p3[0] - p2[0], 2) + pow(p3[1] - p2[1], 2)) + sqrt(pow(p1[0] - p4[0], 2) + pow(p1[1] - p4[1], 2))) / 2;
	int height = (int)(sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)) + sqrt(pow(p3[0] - p4[0], 2) + pow(p3[1] - p4[1], 2))) / 2;
	Vec2d d1(0, 0), d2(0, height), d3(width, height), d4(width, 0);
	vector<Vec2d> src, dst;
	src.push_back(p1);
	src.push_back(p2);
	src.push_back(p3);
	src.push_back(p4);
	dst.push_back(d1);
	dst.push_back(d2);
	dst.push_back(d3);
	dst.push_back(d4);

	Mat homo = getHomo(dst, src);
	Mat patch(height, width, image.type(), Scalar(0, 0, 255));
	std::cout << "width = " << patch.cols << "\n";
	std::cout << "height = " << patch.rows << "\n";
	getTexture(patch, image, homo);
	std::cout << "finish getting texture\n";

	cv::imwrite("1-texture1.jpg", patch);
	cv::imshow("Patch", patch);
	cv::waitKey(0);

	return 0;
}