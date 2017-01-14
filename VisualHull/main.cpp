#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <queue>

// 用于判断投影是否在visual hull内部
struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;  
	cv::Mat m_image;
	const uint m_threshold = 125;

	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);  //P在该相机上的投影像素坐标(u,v,1)

		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];    //齐次坐标三维转二维

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;       //判断点有没有在轮廓(Silhouette)内部，但这只相当于轮廓是矩形啊
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
	}    //这个阈值我不知道是做什么用的
};

// 用于index和实际坐标之间的转换
struct CoordinateInfo
{
	int m_resolution;
	double m_min;
	double m_max;

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;   //不懂这个公式的原理
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};

class Model
{
public:
	typedef std::vector<std::vector<bool>> Pixel;    //二维像素用一个二维矩阵表示?里面还全是0,1
	typedef std::vector<Pixel> Voxel;				 //体素

	Model(int resX = 100, int resY = 100, int resZ = 100);
	~Model();

	void saveModel(const char* pFileName);    //这个不用动
	void saveModelWithNormal(const char* pFileName);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);
	bool judgeInner(int &indexX, int &indexY, int &indexZ);
	bool judgeSurface(int &indexX, int &indexY, int &indexZ);
	void BFS(int &indexX, int &indexY, int &indexZ);
private:
	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	std::vector<Eigen::Vector3f> neiborList;   //getnormal和BFS专用
	std::vector<Eigen::Vector3f> innerList;

	int m_neiborSize;

	std::vector<Projection> m_projectionList;

	Voxel m_voxel;
	Voxel m_surface;
};

Model::Model(int resX, int resY, int resZ)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)
{
	if (resX > 100)
		m_neiborSize = resX / 100;
	else
		m_neiborSize = 1;
	m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, false)));
	//不是很懂这个体素的生成方式
	m_surface = m_voxel;
}

Model::~Model()
{
}

void Model::saveModel(const char* pFileName)     
{
	std::ofstream fout(pFileName);

	bool flag = false;
	int indexX, indexY, indexZ;
	for (indexX = 0; indexX < m_corrX.m_resolution && (!flag); indexX++)
		for (indexY = 0; indexY < m_corrY.m_resolution && (!flag); indexY++)
			for (indexZ = 0; indexZ < m_corrZ.m_resolution && (!flag); indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);//转换成空间实际坐标
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
					flag = true;
				}
	int *coor = new int[3];
	coor[0] = indexX;
	coor[1] = indexY;
	coor[2] = indexZ;
	std::queue<int *>q;
	q.push(coor);
	while (!q.empty())
	{
		indexX = q.front()[0];
		indexY = q.front()[1];
		indexZ = q.front()[2];
		for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
			for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
				for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
				{
					if (!dX && !dY && !dZ)
						continue;
					int neiborX = indexX + dX;
					int neiborY = indexY + dY;
					int neiborZ = indexZ + dZ;
					if (judgeSurface(neiborX, neiborY, neiborZ))
					{
						fout << m_corrX.index2coor(neiborX) << ' ' << m_corrY.index2coor(neiborY) << ' ' << m_corrZ.index2coor(neiborZ) << std::endl; 
						coor = new int[3];
						coor[0] = neiborX;
						coor[1] = neiborY;
						coor[2] = neiborZ;
						q.push(coor);
					}
				}
		q.pop();
	}
}

void Model::saveModelWithNormal(const char* pFileName)
{
	std::ofstream fout(pFileName);

	double midX = m_corrX.index2coor(m_corrX.m_resolution / 2);   //这3个貌似没有用上  
	double midY = m_corrY.index2coor(m_corrY.m_resolution / 2);
	double midZ = m_corrZ.index2coor(m_corrZ.m_resolution / 2);

	bool flag = false;
	int indexX, indexY, indexZ;
	for (indexX = 0; indexX < m_corrX.m_resolution && (!flag); indexX++)
		for (indexY = 0; indexY < m_corrY.m_resolution && (!flag); indexY++)
			for (indexZ = 0; indexZ < m_corrZ.m_resolution && (!flag); indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);//转换成空间实际坐标
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
					flag = true;
					Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ); //只根据一个点的坐标如何求法向量？纳闷
					fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
				}
	int *coor = new int[3];
	coor[0] = indexX;
	coor[1] = indexY;
	coor[2] = indexZ;
	std::queue<int *>q;
	q.push(coor);
	while (!q.empty())
	{
		indexX = q.front()[0];
		indexY = q.front()[1];
		indexZ = q.front()[2];
		for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
			for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
				for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
				{
					if (!dX && !dY && !dZ)
						continue;
					int neiborX = indexX + dX;
					int neiborY = indexY + dY;
					int neiborZ = indexZ + dZ;
					if (judgeSurface(neiborX, neiborY, neiborZ))
					{
						fout << m_corrX.index2coor(neiborX) << ' ' << m_corrY.index2coor(neiborY) << ' ' << m_corrZ.index2coor(neiborZ) << std::endl;
						Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ); //只根据一个点的坐标如何求法向量？纳闷
						fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
						coor = new int[3];
						coor[0] = neiborX;
						coor[1] = neiborY;
						coor[2] = neiborZ;
						q.push(coor);
					}
				}
		q.pop();
	}
}

// 读取相机的内外参数，不用管
void Model::loadMatrix(const char* pFileName)  
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);
	}
}   

       // 读取投影图片，不用动     文件名               前缀                 后缀
void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();
	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	for (int i = 0; i < fileCount; i++)
	{
		std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
		m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
	}
}

//		  得到Voxel模型
void Model::getModel()
{
	int prejectionCount = m_projectionList.size();

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				for (int i = 0; i < prejectionCount; i++)
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);    
					m_voxel[indexX][indexY][indexZ] = m_voxel[indexX][indexY][indexZ] && m_projectionList[i].checkRange(coorX, coorY, coorZ);
				}
}

//判断是否是里点(包括表面点)
bool Model::judgeInner(int &indexX, int &indexY, int &indexZ)   
{
	if (indexX < 0 || indexX >= m_corrX.m_resolution || indexY < 0 || indexY >= m_corrY.m_resolution || indexZ < 0 || indexZ >= m_corrZ.m_resolution)
		return false;
	double coorX = m_corrX.index2coor(indexX);
	double coorY = m_corrY.index2coor(indexY);
	double coorZ = m_corrZ.index2coor(indexZ);
	for (int j = 0; j < m_projectionList.size(); j++)
		m_voxel[indexX][indexY][indexZ] = m_voxel[indexX][indexY][indexZ] && m_projectionList[j].checkRange(coorX, coorY, coorZ);
	return m_voxel[indexX][indexY][indexZ];
}

bool Model::judgeSurface(int &indexX, int &indexY, int &indexZ)
{
	if (indexX < 0 || indexX >= m_corrX.m_resolution || indexY < 0 || indexY >= m_corrY.m_resolution || indexZ < 0 || indexZ >= m_corrZ.m_resolution)
		return false;
	if (!judgeInner(indexX, indexY, indexZ))
	{
		m_surface[indexX][indexY][indexZ] = false;
		return false;
	}
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};
	bool ans = false;
	for (int i = 0; i < 6&&(!ans); i++)
	{
		ans = ans || outOfRange(indexX + dx[i], indexY + dy[i], indexZ + dz[i]);
		double coorX = m_corrX.index2coor(indexX + dx[i]);
		double coorY = m_corrY.index2coor(indexY + dy[i]);
		double coorZ = m_corrZ.index2coor(indexZ + dz[i]);
		for (int j = 0; j < m_projectionList.size(); j++)
			ans = ans || !m_projectionList[j].checkRange(coorX, coorY, coorZ);
	}
	m_surface[indexX][indexY][indexZ] = ans;
	return ans;
}

void Model::BFS(int &indexX, int &indexY, int &indexZ)
{
	bool ***checked = new bool**[m_corrX.m_resolution];
	for (int i = 0; i < m_corrX.m_resolution; i++)
	{
		checked[i] = new bool *[m_corrY.m_resolution];
		for (int j = 0; j < m_corrY.m_resolution; j++)
		{
			checked[i][j] = new bool[m_corrY.m_resolution];
			for (int k = 0; k < m_corrZ.m_resolution; k++)
				checked[i][j][k] = false;
		}
	}
	int *coor=new int[3];
	coor[0] = indexX;
	coor[1] = indexY;
	coor[2] = indexZ;
	std::queue<int *>q;
	q.push(coor);
	while (!q.empty())
	{
		indexX = q.front()[0];
		indexY = q.front()[1];
		indexZ = q.front()[2];
		for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
			for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
				for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
				{
					if (!dX && !dY && !dZ)
						continue;
					int neiborX = indexX + dX;
					int neiborY = indexY + dY;
					int neiborZ = indexZ + dZ;
					if (!checked[neiborX][neiborY][neiborZ])
					{
						if (judgeSurface(neiborX, neiborY, neiborZ))
						{
							neiborList.push_back(Eigen::Vector3f(m_corrX.index2coor(neiborX), m_corrY.index2coor(neiborY), m_corrZ.index2coor(neiborZ)));  //在面上就放到面的vector里面
							coor = new int[3];
							coor[0] = neiborX;
							coor[1] = neiborY;
							coor[2] = neiborZ;
							q.push(coor);
						}
						else if (judgeInner(neiborX, neiborY, neiborZ))
							innerList.push_back(Eigen::Vector3f(m_corrX.index2coor(neiborX), m_corrY.index2coor(neiborY), m_corrZ.index2coor(neiborZ)));   //在体里就放到体的vector里面
						checked[neiborX][neiborY][neiborZ] = true;
					}
				}
		q.pop();
	}
}

void Model::getSurface()
{
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ){
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};
	bool ans = false;
	int indexX, indexY, indexZ;
	for (indexX = m_corrX.m_resolution/2; indexX < m_corrX.m_resolution&&(!ans); indexX++)
		for (indexY = m_corrY.m_resolution/2; indexY < m_corrY.m_resolution&&(!ans); indexY++)
			for (indexZ = m_corrZ.m_resolution/2; indexZ < m_corrZ.m_resolution && (!ans); indexZ++)
			{
				if (!judgeInner(indexX,indexY,indexZ))
				{
					m_surface[indexX][indexY][indexZ] = false;
					continue;
				}
				
				for (int i = 0; i < 6; i++)
				{
					ans = ans || outOfRange(indexX + dx[i], indexY + dy[i], indexZ + dz[i])
						|| !m_voxel[indexX + dx[i]][indexY + dy[i]][indexZ + dz[i]];
				}
				m_surface[indexX][indexY][indexZ] = ans;
			}
	BFS(indexX, indexY, indexZ);
}

Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	//auto outOfRange = [&](int indexX, int indexY, int indexZ){   //判断是否超出分辨率？？
	//	return indexX < 0 || indexY < 0 || indexZ < 0
	//		|| indexX >= m_corrX.m_resolution
	//		|| indexY >= m_corrY.m_resolution
	//		|| indexZ >= m_corrZ.m_resolution;
	//};

	//

	//for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
	//	for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
	//		for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
	//		{
	//			if (!dX && !dY && !dZ)
	//				continue;
	//			int neiborX = indX + dX;
	//			int neiborY = indY + dY;
	//			int neiborZ = indZ + dZ;
	//			if (!outOfRange(neiborX, neiborY, neiborZ))
	//			{
	//				float coorX = m_corrX.index2coor(neiborX);
	//				float coorY = m_corrY.index2coor(neiborY);
	//				float coorZ = m_corrZ.index2coor(neiborZ);   
	//				if (m_surface[neiborX][neiborY][neiborZ])
	//					neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));  //在面上就放到面的vector里面
	//				else if (m_voxel[neiborX][neiborY][neiborZ])
	//					innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));   //在体里就放到体的vector里面
	//			}
	//		}

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;    //差向量
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());//矩阵乘以它的转置
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();  //求了行列式的值？为什么是三维的？
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);   //这个怎么得到的法向量?
	
	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;    //看法向量指内还是指外
	return normalVector;
}

int main(int argc, char** argv)
{
	clock_t t = clock();

	// 分别设置xyz方向的Voxel分辨率
	Model model(300, 300, 300);

	// 读取相机的内外参数
	model.loadMatrix("../../calibParamsI.txt");

	// 读取投影图片
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");

	// 得到Voxel模型
	/*model.getModel();
	std::cout << "get model done\n";*/

	// 获得Voxel模型的表面
	model.getSurface();
	std::cout << "get surface done\n";

	// 将模型导出为xyz格式
	model.saveModel("../../WithoutNormal.xyz");
	std::cout << "save without normal done\n";

	model.saveModelWithNormal("../../WithNormal.xyz");
	std::cout << "save with normal done\n";

	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	std::cout << "save mesh.ply done\n";

	t = clock() - t;
	std::cout << "time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	return (0);
}