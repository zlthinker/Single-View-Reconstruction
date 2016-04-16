#include "svm.h"

int main()
{
	cv::Mat src = cv::imread("box.bmp");
	std::vector<CvPoint> point_vector;
	point_vector.push_back(cv::Point2d(162, 227));
	point_vector.push_back(cv::Point2d(174, 366));
	point_vector.push_back(cv::Point2d(391, 542));
	point_vector.push_back(cv::Point2d(393, 399));
	point_vector.push_back(cv::Point2d(606, 431));
	point_vector.push_back(cv::Point2d(618, 290));
	point_vector.push_back(cv::Point2d(379, 139));
	point_vector.push_back(cv::Point2d(382, 277));

	std::vector<Vec3d> points_2d;
	for (int i = 0; i < point_vector.size(); i++)
	{
		cv::Vec3d pt(point_vector[i].x, point_vector[i].y, 1);
		points_2d.push_back(pt);
	}

	cv::Vec3d origin(point_vector[2].x, point_vector[2].y, 1);
	cv::Vec3d reference(point_vector[3].x, point_vector[3].y, 1);

	float w = (src.cols + src.rows) / 4.0;
	double scale_x = 300;
	double scale_y = 400;
	double scale_z = 183;

	std::vector<CvPoint> parallel_x;
	parallel_x.push_back(point_vector[2]);
	parallel_x.push_back(point_vector[4]);
	parallel_x.push_back(point_vector[3]);
	parallel_x.push_back(point_vector[5]);
	parallel_x.push_back(point_vector[0]);
	parallel_x.push_back(point_vector[6]);
	std::vector<CvPoint> parallel_y;
	parallel_y.push_back(point_vector[2]);
	parallel_y.push_back(point_vector[1]);
	parallel_y.push_back(point_vector[3]);
	parallel_y.push_back(point_vector[0]);
	parallel_y.push_back(point_vector[5]);
	parallel_y.push_back(point_vector[6]);
	std::vector<CvPoint> parallel_z;
	parallel_z.push_back(point_vector[1]);
	parallel_z.push_back(point_vector[0]);
	parallel_z.push_back(point_vector[2]);
	parallel_z.push_back(point_vector[3]);
	parallel_z.push_back(point_vector[4]);
	parallel_z.push_back(point_vector[5]);

	//center image
	for (std::vector<int>::size_type i = 0; i != parallel_x.size(); ++i){
		parallel_x[i].x -= src.cols / 2;
		parallel_x[i].y -= src.rows / 2;
	}

	for (std::vector<int>::size_type i = 0; i != parallel_y.size(); ++i){
		parallel_y[i].x -= src.cols / 2;
		parallel_y[i].y -= src.rows / 2;
	}

	for (std::vector<int>::size_type i = 0; i != parallel_z.size(); ++i){
		parallel_z[i].x -= src.cols / 2;
		parallel_z[i].y -= src.rows / 2;
	}

	CvPoint3D32f x_vanish = calc_vanishing_point(parallel_x, w);
	CvPoint3D32f y_vanish = calc_vanishing_point(parallel_y, w);
	CvPoint3D32f z_vanish = calc_vanishing_point(parallel_z, w);
	cv::Vec3d v_x(x_vanish.x + src.cols / 2, x_vanish.y + src.rows / 2, 1);
	cv::Vec3d v_y(y_vanish.x + src.cols / 2, y_vanish.y + src.rows / 2, 1);
	cv::Vec3d v_z(z_vanish.x + src.cols / 2, z_vanish.y + src.rows / 2, 1);

	std::cout << "vanishing x: " << v_x << "\n";
	std::cout << "vanishing y: " << v_y << "\n";
	std::cout << "vanishing z: " << v_z << "\n\n";

	std::vector<cv::Vec3d> src_point_vector;
	std::vector<cv::Vec3d> target_point_vector;

	src_point_vector.push_back(points_2d[2]);
	src_point_vector.push_back(points_2d[4]);
	src_point_vector.push_back(points_2d[7]);
	src_point_vector.push_back(points_2d[1]);
	target_point_vector.push_back(cv::Vec3d(0, 0, 1));
	target_point_vector.push_back(cv::Vec3d(scale_x, 0, 1));
	target_point_vector.push_back(cv::Vec3d(scale_x, scale_y, 1));
	target_point_vector.push_back(cv::Vec3d(0, scale_y, 1));
	cv::Mat Hxy = getHomo(src_point_vector, target_point_vector);

	// calculate scale of z axis
	double z_temp = cal3dZ(v_x, v_y, v_z, reference, origin, origin, 1);
	double scale_z_true = scale_z / z_temp;

	cv::Vec3d V1 = cal3dXYZ(v_x, v_y, v_z, points_2d[0], points_2d[1], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V2 = cal3dXYZ(v_x, v_y, v_z, points_2d[1], points_2d[1], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V3 = cal3dXYZ(v_x, v_y, v_z, points_2d[2], points_2d[2], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V4 = cal3dXYZ(v_x, v_y, v_z, points_2d[3], points_2d[2], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V5 = cal3dXYZ(v_x, v_y, v_z, points_2d[4], points_2d[4], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V6 = cal3dXYZ(v_x, v_y, v_z, points_2d[5], points_2d[4], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V7 = cal3dXYZ(v_x, v_y, v_z, points_2d[6], points_2d[7], points_2d[2], scale_z_true, Hxy);
	cv::Vec3d V8 = cal3dXYZ(v_x, v_y, v_z, points_2d[7], points_2d[7], points_2d[2], scale_z_true, Hxy);

	std::cout << "3D coordinates:\n";
	std::cout << V1 << "\n";
	std::cout << V2 << "\n";
	std::cout << V3 << "\n";
	std::cout << V4 << "\n";
	std::cout << V5 << "\n";
	std::cout << V6 << "\n";
	std::cout << V7 << "\n";
	std::cout << V8 << "\n\n";


	/******** get texture here **********/
	double text_x = scale_x; // distance2d(points_2d[2], points_2d[4]);
	double text_y = scale_y; // distance2d(points_2d[2], points_2d[1]);
	double text_z = scale_z; // distance2d(points_2d[2], points_2d[3]);
	std::vector<cv::Vec3d> src_text_vector;
	std::vector<cv::Vec3d> target_text_vector;
	
	// XY
	std::cout << "Save texture XY...\n";
	src_text_vector.push_back(cv::Vec3d(point_vector[6].x, point_vector[6].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[0].x, point_vector[0].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[3].x, point_vector[3].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[5].x, point_vector[5].y, 1));
	target_text_vector.push_back(cv::Vec3d(0, 0, 1));
	target_text_vector.push_back(cv::Vec3d(0, text_x, 1));
	target_text_vector.push_back(cv::Vec3d(text_y, text_x, 1));
	target_text_vector.push_back(cv::Vec3d(text_y, 0, 1));

	cv::Mat text_xy = getHomo(target_text_vector, src_text_vector);
	cv::Mat texture_xy(text_x, text_y, src.type(), Scalar(0, 0, 0));
	getTexture(texture_xy, src, text_xy);
	cv::imwrite("box_texture_xy.jpg", texture_xy);

	// XZ
	std::cout << "Save texture XZ...\n";
	src_text_vector.clear();
	target_text_vector.clear();
	src_text_vector.push_back(cv::Vec3d(point_vector[3].x, point_vector[3].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[2].x, point_vector[2].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[4].x, point_vector[4].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[5].x, point_vector[5].y, 1));
	target_text_vector.push_back(cv::Vec3d(0, 0, 1));
	target_text_vector.push_back(cv::Vec3d(0, text_z, 1));
	target_text_vector.push_back(cv::Vec3d(text_x, text_z, 1));
	target_text_vector.push_back(cv::Vec3d(text_x, 0, 1));

	cv::Mat text_xz = getHomo(target_text_vector, src_text_vector);
	cv::Mat texture_xz(text_z, text_x, src.type(), Scalar(0, 0, 0));
	getTexture(texture_xz, src, text_xz);
	cv::imwrite("box_texture_xz.jpg", texture_xz);

	// YZ
	std::cout << "Save texture YZ...\n";
	src_text_vector.clear();
	target_text_vector.clear();
	src_text_vector.push_back(cv::Vec3d(point_vector[0].x, point_vector[0].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[1].x, point_vector[1].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[2].x, point_vector[2].y, 1));
	src_text_vector.push_back(cv::Vec3d(point_vector[3].x, point_vector[3].y, 1));
	target_text_vector.push_back(cv::Vec3d(0, 0, 1));
	target_text_vector.push_back(cv::Vec3d(0, text_z, 1));
	target_text_vector.push_back(cv::Vec3d(text_y, text_z, 1));
	target_text_vector.push_back(cv::Vec3d(text_y, 0, 1));

	cv::Mat text_yz = getHomo(target_text_vector, src_text_vector);
	cv::Mat texture_yz(text_z, text_y, src.type(), Scalar(0, 0, 0));
	getTexture(texture_yz, src, text_yz);
	cv::imwrite("box_texture_yz.jpg", texture_yz);

	// write VRML model
	std::cout << "Save VRML file...\n";
	std::ofstream fout("out.wrl");
	std::string textures[] = { "box_texture_xy.jpg", "box_texture_xz.jpg", "box_texture_yz.jpg" };
	std::vector<cv::Vec3d> text_points;
	text_points.push_back(V1);
	text_points.push_back(V7);
	text_points.push_back(V6);
	text_points.push_back(V4);
	text_points.push_back(V3);
	text_points.push_back(V4);
	text_points.push_back(V6);
	text_points.push_back(V5);
	text_points.push_back(V2);
	text_points.push_back(V1);
	text_points.push_back(V4);
	text_points.push_back(V3);

	create_crml_file(text_points, fout, textures);

	return 0;
}