#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include "eigen.h"

#include <cv.h>
#include <highgui.h>

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
  
int main(int argc,char** argv)  
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

	//calc_H(point_vector, target_point_vector, matrix_H);
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
CvPoint3D32f calc_vanishing_point(std::vector<CvPoint> points, float w)
{
	int num_lines = points.size()/2;
	//compute lines
	CvPoint3D32f* lines = (CvPoint3D32f *)malloc(num_lines*sizeof(CvPoint3D32f));
	for(std::vector<int>::size_type i=0;i!=num_lines;++i)
	{
		lines[i].x = points[2 * i].y/w - points[2 * i + 1].y/w;
		lines[i].y = points[2 * i + 1].x/w - points[2 * i].x/w;
		lines[i].z = (points[2 * i].x * points[2 * i + 1].y - points[2 * i + 1].x * points[2 * i].y) / (w * w);
	}
	

	//compute m
	float *eig_val = (float *) malloc (3*sizeof(float));
	float **eig_vec = (float**) malloc (3*sizeof(float*));
	float **m = (float**) malloc (3*sizeof(float*));
	for (int i=0;i<3;i++) {
		m[i] = (float *) malloc (3*sizeof(float));
		eig_vec[i] = (float *) malloc (3*sizeof(float));
	}
	for (int i=0;i<3;i++) {
		for (int j = 0; j < 3; j++)
			m[i][j] = 0.0;
	}


	for(std::vector<int>::size_type i=0;i!=num_lines;++i)
	{
		add_point(m,&lines[i]);
	}
	
	//compute Vertice
	eig_sys (3,m,eig_vec,eig_val);

	float min_eigen = 9999999.9;
	int index = -1;
	for ( int i = 0; i < 3; i++ )
		if (eig_val[i] < min_eigen)
		{
			min_eigen = eig_val[i];
			index = i;
		}
	float scale = w/eig_vec[index][2];
	return cvPoint3D32f(eig_vec[index][0]*scale, eig_vec[index][1]*scale, eig_vec[index][2]*scale) ;
}

void add_point(float **m,CvPoint3D32f * p)
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

/*
int calc_H(std::vector<CvPoint> point_vector, std::vector<CvPoint> target_point_vector, float * matrix_H ){
	
	if ( point_vector.size() != 4 || target_point_vector.size() != 4 )
		return -1;
	float m[8][9] = { 0.0 };
	float m_T[9][8] = { 0.0 };

	for (std::vector<int>::size_type i=0;i!= point_vector.size();++i){
		m[2*i][0] = point_vector[i].x;m[2*i][1] = point_vector[i].y;m[2*i][2] = 1 ;m[2*i][3] = 0 ;m[2*i][4] = 0;m[2*i][5] = 0;
		m[2*i][6] = -target_point_vector[i].x * point_vector[i].x;m[2*i][7] = -target_point_vector[i].x * point_vector[i].y ;m[2*i][8] = -target_point_vector[i].x;
		m[2*i+1][0] = 0 ;m[2*i+1][1] = 0;m[2*i+1][2] = 0;m[2*i+1][3] = point_vector[i].x;m[2*i+1][4] = point_vector[i].y;m[2*i+1][5] = 1 ;
		m[2*i+1][6] = -target_point_vector[i].y * point_vector[i].x;m[2*i+1][7] = -target_point_vector[i].y * point_vector[i].y ;m[2*i+1][8] = -target_point_vector[i].y;
	}
	for ( int i = 0; i < 9; i++ )
		for ( int j = 0; j < 8; j++){
			m_T[i][j] = m[j][i];
		}

	//eigendecompostion
	float *eig_val = (float *) malloc (9*sizeof(float));
	float **eig_vec = (float**) malloc (9*sizeof(float*));
	float **prod = (float**) malloc (9*sizeof(float*));
	for (int i=0;i<9;i++) {
		prod[i] = (float *) malloc (9*sizeof(float));
		eig_vec[i] = (float *) malloc (9*sizeof(float));
	}

	float sum = 0.0;
	//product of m and m_T
	for ( int i = 0; i < 9; i++ )
		for ( int j = 0; j < 9; j++){
			sum = 0.0;
			for ( int k = 0; k < 8; k++){
				sum += m_T[i][k] * m[k][j];
			}
			prod[i][j] = sum;
		}

	eig_sys (9,prod,eig_vec,eig_val);

	float min_eigen = 9999999.9;
	int index = -1;
	for ( int i = 0; i < 9; i++ )
		if (eig_val[i] < min_eigen)
		{
			min_eigen = eig_val[i];
			index = i;
		}
	matrix_H = eig_vec[index];
	return 0;
}	
*/