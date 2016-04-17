#include "svm.h"


void my_callback(int event,int x,int y,int flags,void* param);  
CvPoint3D32f calc_vanishing_point(std::vector<CvPoint> points, float w);
void add_point(float **m,CvPoint3D32f * p);
int calc_H(std::vector<CvPoint> point_vector, std::vector<CvPoint> target_point_vector, float * matrix_H );

CvPoint pt1=cvPoint(0,0);  
CvPoint pt2=cvPoint(0,0);  
bool drawing_line = false;

bool drawing_point = false;
std::vector<CvPoint> point_vector;
std::vector<CvPoint> x_point_vector;
std::vector<CvPoint> y_point_vector;
std::vector<CvPoint> z_point_vector;
  
void draw_line(IplImage* img,CvPoint p1,CvPoint p2)  
{  
    cvLine(  
        img,  
        p1,  
        p2,  
        CV_RGB(255,0,0),  
        3,  
        8);  
}  
  
int main(int argc,char** argv)  
{  
	cv::Mat src = cv::imread("input.jpg");
    IplImage* img=cvLoadImage("input.jpg");
    IplImage* image = cvCloneImage(img);
    IplImage* temp=cvCloneImage(image);  
    cvNamedWindow("Draw Line Example", CV_WINDOW_NORMAL);  
    cvSetMouseCallback(  
        "Draw Line Example",  
        my_callback,  
        (void*)image);  

    while(1)  
    {  
        cvCopyImage(image,temp);  
        if(drawing_line) draw_line(temp,pt1,pt2);  
        cvShowImage("Draw Line Example",temp); 
		int c = cv::waitKey(20);
		if ( c == 'q' )
		{
			break;
		
		}
        switch(c){
		case 'x':
			point_vector.clear();
			pt1 = cvPoint(0,0);
			pt2 = cvPoint(0,0);
			cvCopyImage(img,image);
			break;
		case 'y':
			x_point_vector = point_vector;
			point_vector.clear();
			pt1 = cvPoint(0,0);
			pt2 = cvPoint(0,0);
			cvCopyImage(img,image); 
			break;
		case 'z':
			y_point_vector = point_vector;
			point_vector.clear();
			pt1 = cvPoint(0,0);
			pt2 = cvPoint(0,0);
			cvCopyImage(img,image); 
			break;
		case 's':
			z_point_vector = point_vector;
			point_vector.clear();
			pt1 = cvPoint(0,0);
			pt2 = cvPoint(0,0);
			cvCopyImage(img,image);
			break;
		case 'h':
			point_vector.clear();
			drawing_point = true;
			cvCopyImage(img,image); 
			break;
		default:
			break;
		}

	}

	float w = (img->width + img->height)/4.0;

	// calculate vanishing points
	for (std::vector<int>::size_type i=0;i!= x_point_vector.size();++i){
		x_point_vector[i].x -= img->width/2;
		x_point_vector[i].y -= img->height/2;
	}

	for (std::vector<int>::size_type i=0;i!= y_point_vector.size();++i){
		y_point_vector[i].x -= img->width/2;
		y_point_vector[i].y -= img->height/2;
	}

	for (std::vector<int>::size_type i=0;i!= z_point_vector.size();++i){
		z_point_vector[i].x -= img->width/2;
		z_point_vector[i].y -= img->height/2;
	}

	CvPoint3D32f x_vanish = calc_vanishing_point(x_point_vector,w);
	CvPoint3D32f y_vanish = calc_vanishing_point(y_point_vector,w);
	CvPoint3D32f z_vanish = calc_vanishing_point(z_point_vector,w);


	point_vector.clear();
	point_vector.push_back(cvPoint(2316, 3681));
	point_vector.push_back(cvPoint(1711, 3372));
	point_vector.push_back(cvPoint(2226, 3300));
	point_vector.push_back(cvPoint(2942, 3570));
	point_vector.push_back(cvPoint(2326, 3092));


	// calculate homography of x-y plane and scale in xyz axis
	if (point_vector.size() < 5)
	{
		std::cout << "Less than 5 points are clicked!\n";
		return -1;
	}

	double scale_x = 300, scale_y = 600, scale_z = 300;
	std::vector <cv::Point2d> target_point_vector;
	target_point_vector.push_back(cv::Point2d(0,0));
	target_point_vector.push_back(cv::Point2d(0, scale_y));
	target_point_vector.push_back(cv::Point2d(scale_x, scale_y));
	target_point_vector.push_back(cv::Point2d(scale_x, 0));

	std::vector <cv::Point2d> src_point_vector;
	src_point_vector.push_back(cv::Point2d(point_vector[0].x, point_vector[0].y));
	src_point_vector.push_back(cv::Point2d(point_vector[1].x, point_vector[1].y));
	src_point_vector.push_back(cv::Point2d(point_vector[2].x, point_vector[2].y));
	src_point_vector.push_back(cv::Point2d(point_vector[3].x, point_vector[3].y));
	cv::Vec3d origin(point_vector[0].x, point_vector[0].y, 1);

	cv::Mat Hxy = cv::findHomography(src_point_vector, target_point_vector);
	cv::Vec3d z_scale(point_vector[4].x, point_vector[4].y, 1);

	point_vector.clear();
	//cv::Vec3d v_x(x_vanish.x + image->width / 2, x_vanish.y + image->height / 2, 1);
	//cv::Vec3d v_y(y_vanish.x + image->width / 2, y_vanish.y + image->height / 2, 1);
	//cv::Vec3d v_z(z_vanish.x + image->width / 2, z_vanish.y + image->height / 2, 1);
	cv::Vec3d v_x(9563.7, 2403.48, 1);
	cv::Vec3d v_y(-86.75, 2409.9, 1);
	cv::Vec3d v_z(5534.12, 530677, 1);

	std::cout << "origin: " << origin << "\n";
	std::cout << "vanishing x: " << v_x << "\n";
	std::cout << "vanishing y: " << v_y << "\n";
	std::cout << "vanishing z: " << v_z << "\n";
	std::cout << "homography: " << Hxy << "\n";

	// calculate scale of z axis
	double z_temp = cal3dZ(v_x, v_y, v_z, z_scale, origin, origin, 1);
	scale_z = scale_z / z_temp;
	
	double x_coord, y_coord, z_coord;
	cv::Vec3d pt0, pt1, pt2, pt3;
	cv::Vec3d top0, bottom0, top1, bottom1, top2, bottom2, top3, bottom3;
	std::vector<cv::Vec3d> src_text_vector;
	std::vector<cv::Vec3d> target_text_vector;
	int cnt = 111;
	
	while (1)
	{
		cvCopyImage(image, temp);
		int c = cv::waitKey(20);
		if (c == 27)
		{
			break;
		}
		if (c == 'p')
		{
			if (point_vector.size() < 4) { break; }
			top0 = cv::Vec3d(point_vector[0].x, point_vector[0].y, 1);
			bottom0 = cv::Vec3d(point_vector[1].x, point_vector[1].y, 1);
			pt0 = cal3dXYZ(v_x, v_y, v_z, top0, bottom0, origin, scale_z, Hxy);

			pt1 = cal3dXYZ(v_x, v_y, v_z, bottom0, bottom0, origin, scale_z, Hxy);

			top1 = cv::Vec3d(point_vector[2].x, point_vector[2].y, 1);
			bottom1 = cv::Vec3d(point_vector[3].x, point_vector[3].y, 1);
			pt3 = cal3dXYZ(v_x, v_y, v_z, top1, bottom1, origin, scale_z, Hxy);

			pt2 = cal3dXYZ(v_x, v_y, v_z, bottom1, bottom1, origin, scale_z, Hxy);
			std::cout << "3D coordinate 0: " << pt0 << "\n";
			std::cout << "3D coordinate 1: " << pt1 << "\n";
			std::cout << "3D coordinate 2: " << pt2 << "\n";
			std::cout << "3D coordinate 3: " << pt3 << "\n";

			double text_w = distance3d(pt0, pt3);
			double text_h = distance3d(pt0, pt1);

			std::cout << "Save texture...\n";
			src_text_vector.clear();
			target_text_vector.clear();
			src_text_vector.push_back(top0);
			src_text_vector.push_back(bottom0);
			src_text_vector.push_back(bottom1);
			src_text_vector.push_back(top1);
			target_text_vector.push_back(cv::Vec3d(0, 0, 1));
			target_text_vector.push_back(cv::Vec3d(0, text_h, 1));
			target_text_vector.push_back(cv::Vec3d(text_w, text_h, 1));
			target_text_vector.push_back(cv::Vec3d(text_w, 0, 1));

			cv::Mat text = getHomo(target_text_vector, src_text_vector);
			std::cout << "texture: " << text_w << ", " << text_h << "\n";
			cv::Mat texture(text_h, text_w, src.type(), Scalar(0, 0, 0));
			getTexture(texture, src, text);
			std::string text_file = std::to_string(cnt) + ".jpg";
			cv::imwrite(text_file, texture);
			cnt++;

			std::string textures[] = { text_file };
			std::cout << textures[0] << std::endl;
			std::vector<cv::Vec3d> text_points;
			text_points.push_back(pt1);
			text_points.push_back(pt0);
			text_points.push_back(pt3);
			text_points.push_back(pt2);
			std::ofstream fout("livingroom.wrl", std::ofstream::out | std::ofstream::app);
			create_crml_file(text_points, fout, textures);
			fout.close();

			point_vector.clear();
		}


		else if (c == 'v')
		{

			if (point_vector.size() < 8) { break; }
			top0 = cv::Vec3d(point_vector[0].x, point_vector[0].y, 1);
			bottom0 = cv::Vec3d(point_vector[1].x, point_vector[1].y, 1);
			pt0 = cal3dXYZ(v_x, v_y, v_z, top0, bottom0, origin, scale_z, Hxy);

			top1 = cv::Vec3d(point_vector[2].x, point_vector[2].y, 1);
			bottom1 = cv::Vec3d(point_vector[3].x, point_vector[3].y, 1);
			pt1 = cal3dXYZ(v_x, v_y, v_z, top1, bottom1, origin, scale_z, Hxy);

			top2 = cv::Vec3d(point_vector[4].x, point_vector[4].y, 1);
			bottom2 = cv::Vec3d(point_vector[5].x, point_vector[5].y, 1);
			pt2 = cal3dXYZ(v_x, v_y, v_z, top2, bottom2, origin, scale_z, Hxy);

			top3 = cv::Vec3d(point_vector[6].x, point_vector[6].y, 1);
			bottom3 = cv::Vec3d(point_vector[7].x, point_vector[7].y, 1);
			pt3 = cal3dXYZ(v_x, v_y, v_z, top3, bottom3, origin, scale_z, Hxy);


			std::cout << "3D coordinate 0: " << pt0 << "\n";
			std::cout << "3D coordinate 1: " << pt1 << "\n";
			std::cout << "3D coordinate 2: " << pt2 << "\n";
			std::cout << "3D coordinate 3: " << pt3 << "\n";

			double text_w = distance3d(pt0, pt3);
			double text_h = distance3d(pt0, pt1);

			std::cout << "Save texture...\n";
			src_text_vector.clear();
			target_text_vector.clear();
			src_text_vector.push_back(top0);
			src_text_vector.push_back(top1);
			src_text_vector.push_back(top2);
			src_text_vector.push_back(top3);
			target_text_vector.push_back(cv::Vec3d(0, 0, 1));
			target_text_vector.push_back(cv::Vec3d(0, text_h, 1));
			target_text_vector.push_back(cv::Vec3d(text_w, text_h, 1));
			target_text_vector.push_back(cv::Vec3d(text_w, 0, 1));

			cv::Mat text = getHomo(target_text_vector, src_text_vector);
			std::cout << "texture: " << text_w << ", " << text_h << "\n";
			cv::Mat texture(text_h, text_w, src.type(), Scalar(0, 0, 0));
			getTexture(texture, src, text);
			std::string text_file = std::to_string(cnt) + ".jpg";
			cv::imwrite(text_file, texture);
			cnt++;

			std::string textures[] = { text_file };
			std::cout << textures[0] << std::endl;
			std::vector<cv::Vec3d> text_points;
			text_points.push_back(pt1);
			text_points.push_back(pt0);
			text_points.push_back(pt3);
			text_points.push_back(pt2);
			std::ofstream fout("livingroom.wrl", std::ofstream::out | std::ofstream::app);
			create_crml_file(text_points, fout, textures);
			fout.close();

			point_vector.clear();
		}
	}

	cvReleaseImage(&img);
    cvReleaseImage(&image);  
    cvReleaseImage(&temp);  
    cvDestroyAllWindows();  

	return 0;
}  
void my_callback(int event,int x,int y,int flags,void* param)  
{  
    IplImage* image=(IplImage*) param;  
    switch (event)  
    {  
    case CV_EVENT_MOUSEMOVE:  
        {  
            if(drawing_line)  
            {  
                pt2=cvPoint(x,y);  
            }  
        }  
        break;  
    case CV_EVENT_LBUTTONDOWN:  
        {  
            pt2=pt1=cvPoint(x,y);  
            drawing_line=true;  
        }  
        break;  
    case CV_EVENT_LBUTTONUP:  
        {  
			std::cout << "Press at (" << x << ", " << y << ")\n";
            drawing_line=false;  
            pt2=cvPoint(x,y);  
            draw_line(image,pt1,pt2);
			point_vector.push_back(pt1);
			if ( !drawing_point )
				point_vector.push_back(pt2);

        }  
        break;  
    default:  
        break;  
    }  
}  
