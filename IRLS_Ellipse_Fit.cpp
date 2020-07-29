#include<iostream>
#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;


struct Ellipse_struct
{
	double A, B, C, D, E, F;
};

//带权重的椭圆拟合
Ellipse_struct IRLS(vector<Point2d>select_points, Mat& weightm, double p)
{
	Mat leftm = Mat_<double>(select_points.size(), 5);
	Mat rightm = Mat_<double>(select_points.size(), 1);

	for (int i = 0; i < select_points.size(); i++)
	{
		leftm.at<double>(i, 0) = select_points[i].x*select_points[i].y;
		leftm.at<double>(i, 1) = select_points[i].y*select_points[i].y;
		leftm.at<double>(i, 2) = select_points[i].x;
		leftm.at<double>(i, 3) = select_points[i].y;
		leftm.at<double>(i, 4) = 1;

		rightm.at<double>(i, 0) = -(select_points[i].x*select_points[i].x);
	}

	Mat cof_mat = (leftm.t()*weightm.t()*weightm*leftm).inv()*leftm.t()*weightm.t()*weightm*rightm;//求解AX=B

	double A, B, C, D, E, F;
	A = 1;
	B = cof_mat.at<double>(0);
	C = cof_mat.at<double>(1);
	D = cof_mat.at<double>(2);
	E = cof_mat.at<double>(3);
	F = cof_mat.at<double>(4);
	Ellipse_struct ellipse_model = { A, B, C, D, E, F };

	//确定下一次迭代时的权重矩阵
	cv::pow(abs(leftm*cof_mat - rightm),(p-2)/2, weightm);//误差计算
	double weight_sum = cv::sum(weightm)[0];
	weightm = weightm / weight_sum;
	weightm = Mat::diag(weightm);

	return ellipse_model;
}

Ellipse_struct MyIRLS(vector<Point2d>select_points, const double p,const int iter_num)
{
	Mat weightm = Mat::eye(Size(select_points.size(), select_points.size()), CV_64F);//创建单位矩阵初始化权重矩阵

	size_t i = 0;
	Ellipse_struct best_ellipse_model;//椭圆模型用于迭代
	while (i<iter_num)
	{
		best_ellipse_model = IRLS(select_points, weightm, p);//bound：点集，weightm：权重矩阵，p：范数
		i++;
	}
	return best_ellipse_model;
}

//获取拟合后的椭圆参数
void GetEllipseParam(Ellipse_struct best_ellipse, double&h, double&k, double&alpha, double&majorAxis, double&minorAxis)
{
	double A, B, C, D, E, F;
	A = best_ellipse.A / best_ellipse.F;
	B = best_ellipse.B / best_ellipse.F;
	C = best_ellipse.C / best_ellipse.F;
	D = best_ellipse.D / best_ellipse.F;
	E = best_ellipse.E / best_ellipse.F;
	F = best_ellipse.F / best_ellipse.F;

	// Center
	h = ((B * E) - (2 * C * D)) / ((4 * A * C) - (B * B));
	k = ((B * D) - (2 * A * E)) / ((4 * A * C) - (B * B));

	// Major and Minor axis radius
	majorAxis = max(sqrt(2 * (A*h*h + C*k*k + B*h*k - 1) / (A + C - sqrt((A - C)*(A - C) + B*B))), sqrt(2 * (A*h*h + C*k*k + B*h*k - 1) / (A + C + sqrt((A - C)*(A - C) + B*B))));
	minorAxis = min(sqrt(2 * (A*h*h + C*k*k + B*h*k - 1) / (A + C - sqrt((A - C)*(A - C) + B*B))), sqrt(2 * (A*h*h + C*k*k + B*h*k - 1) / (A + C + sqrt((A - C)*(A - C) + B*B))));
	// Rotation
	alpha = (-atan(B / (A - C))) / 2;

}

//读取txt文件中的边缘点
Point2d SubTemp(string&temp_col, string&temp_row)
{
	Point2d bound;
	int n;
	istringstream is_c(temp_col);
	istringstream is_r(temp_row);
	string scols, srows;
	is_c >> n >> scols;
	is_r >> n >> srows;
	double fcols = stof(scols);
	double frows = stof(srows);
	bound.x = fcols;
	bound.y = frows;
	return bound;
}

int main()
{
	ifstream myfile_c("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5_col.txt");
	ifstream myfile_r("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5_row.txt");

	vector<Point2d>bound;//亚像素边缘点集
	//读取亚像素点
	string temp_c;
	string temp_r;
	if (!myfile_c.is_open() || !myfile_r.is_open())
	{
		cout << "未成功打开文件" << endl;
	}
	while (getline(myfile_c, temp_c) && getline(myfile_r, temp_r))
	{
		Point2d bound_points = SubTemp(temp_c, temp_r);
		bound.push_back(bound_points);
	}
	myfile_c.close();
	myfile_r.close();

	Ellipse_struct best_ellipse_model;//椭圆模型用于迭代

	best_ellipse_model = MyIRLS(bound, 2, 50);

	double x, y, phi, a, b;
	GetEllipseParam(best_ellipse_model, x, y, phi, a, b);
	return 0;
}