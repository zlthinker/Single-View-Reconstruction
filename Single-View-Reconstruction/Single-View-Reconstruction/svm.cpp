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
        1,  
        8);  
}  
  
int svm(int argc,char** argv)  
{  
    IplImage* img=cvLoadImage("1.jpg");
    IplImage* image = cvCloneImage(img);
    IplImage* temp=cvCloneImage(image);  
    cvNamedWindow("Draw Line Example");  
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
		if ( c == 27 )
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

	// calculate homography of x-y plane and scale in xyz axis
	if (point_vector.size() < 5)
	{
		std::cout << "Less than 5 points are clicked!\n";
		return -1;
	}
	std::vector <cv::Point2d> target_point_vector;
	target_point_vector.push_back(cv::Point2d(0,0));
	target_point_vector.push_back(cv::Point2d(0, 1));
	target_point_vector.push_back(cv::Point2d(1, 1));
	target_point_vector.push_back(cv::Point2d(1, 0));

	std::vector <cv::Point2d> src_point_vector;
	src_point_vector.push_back(cv::Point2d(point_vector[0].x, point_vector[0].y));
	src_point_vector.push_back(cv::Point2d(point_vector[1].x, point_vector[1].y));
	src_point_vector.push_back(cv::Point2d(point_vector[2].x, point_vector[2].y));
	src_point_vector.push_back(cv::Point2d(point_vector[3].x, point_vector[3].y));
	cv::Vec3d origin(point_vector[0].x, point_vector[0].y, 1);

	cv::Mat Hxy = cv::findHomography(src_point_vector, target_point_vector);
	cv::Vec3d z_scale(point_vector[4].x, point_vector[4].y, 1);

	point_vector.clear();
	cv::Vec3d v_x(x_vanish.x + image->width / 2, x_vanish.y + image->height / 2, 1);
	cv::Vec3d v_y(y_vanish.x + image->width / 2, y_vanish.y + image->height / 2, 1);
	cv::Vec3d v_z(z_vanish.x + image->width / 2, z_vanish.y + image->height / 2, 1);

	std::cout << "origin: " << origin << "\n";
	std::cout << "vanishing x: " << v_x << "\n";
	std::cout << "vanishing y: " << v_y << "\n";
	std::cout << "vanishing z: " << v_z << "\n";
	std::cout << "homography: " << Hxy << "\n";

	// calculate scale of z axis
	double z_temp = cal3dZ(v_x, v_y, v_z, z_scale, origin, origin, 1);
	double scale_z = 1 / z_temp;
	
	double x_coord, y_coord, z_coord;
	cv::Vec3d top, bottom;
	
	while (1)
	{
		cvCopyImage(image, temp);
		if (drawing_line) draw_line(temp, pt1, pt2);
		cvShowImage("Draw Line Example", temp);
		int c = cv::waitKey(20);
		if (c == 27)
		{
			break;

		}
		switch (c){
		case 'p':
			top = cv::Vec3d(point_vector[0].x, point_vector[0].y, 1);
			bottom = cv::Vec3d(point_vector[1].x, point_vector[1].y, 1);
			z_coord = cal3dZ(v_x, v_y, v_z, top, bottom, origin, scale_z);
			cal3dXY(bottom, Hxy, x_coord, y_coord);
			std::cout << "3D coordinate: (" << x_coord << ", " << y_coord << ", " << z_coord << ")\n";
			point_vector.clear();

			break;
		default:
			break;
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
