#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>

#define DEBUG

using namespace std;

// 数学常数
const double INF = 1.7e308;
const double EPS = 1e-6;
const double E = 2.718281828459;

// --- 超参数 ---
#ifdef DEBUG
const int TRAIN_NUM = 100;
const int TEST_NUM = 50;
#else
const int TRAIN_NUM = 60000;
const int TEST_NUM = 10000;
#endif
const double TARGET_TRAIN_CORRECTNESS = 0.95;
const int ITERATION_CAP = 300; // 迭代次数上限
const double LEARNING_RATE = 0.5; // 梯度下降学习率
// --- 超参数 ---

// --- 网络结构 ---
// 输入层: 28*28
// 隐藏层: 20
// 输出层: 10
const int LAYER_NUM = 3;
const int IMAGE_ROW = 28;
const int IMAGE_COL = 28;
const int IMAGE_SIZE = IMAGE_ROW * IMAGE_COL;
const int LABEL_SIZE = 1;
const int OUT_SIZE = 10;
const int NEURON_NUM[LAYER_NUM] = {IMAGE_SIZE, 20, OUT_SIZE};
// --- 网络结构 ---

//激活函数
inline double sigmoid(double x) {
    return 1.0 / (1 + pow(E, -x)); //BUG,分母还有个1+
}

inline double ReLU(double x) {
    return x > 0 ? x : 0;
}

struct Matrix {

    vector<vector<double>> mat;

    Matrix() {
        mat.resize(0);
    }

    Matrix(int row, int col, double val = 0.0) {
        resize(row, col, val);
    }

    int row() const {
        return mat.size();
    }

    int col() const {
        return mat[0].size();
    }

    void clear() {
        for (int i = 0; i < row(); i++)
            for (int j = 0; j < col(); j++)
                mat[i][j] = 0.0;
    }

    void resize(int row, int col, double val = 0.0) {
        mat.resize(row);
        for (int i = 0; i < row; i++)
            mat[i].resize(col, val);
    }

    void resize(const Matrix &y) {
        resize(y.row(), y.col());
    }

    const vector<double> &operator[](int i) const {
        return mat[i];
    }

    vector<double> &operator[](int i) {
        return mat[i];
    }

    Matrix &operator=(const Matrix &a) {
        this->resize(a);
        for (int i = 0; i < a.row(); i++)
            for (int j = 0; j < a.col(); j++)
                mat[i][j] = a.mat[i][j];
        return *this;
    }

    Matrix operator+(const Matrix &a) const {
        if (this->row() != a.row() || this->col() != a.col())
            throw "ERROR: Matrix addition format wrong";
        Matrix res;
        res.resize(a.row(), a.col());
        for (int i = 0; i < a.row(); i++)
            for (int j = 0; j < a.col(); j++)
                res[i][j] = mat[i][j] + a.mat[i][j];
        return res;
    }

    Matrix operator-(const Matrix &b) const {
        if (this->row() != b.row() || this->col() != b.col())
            throw "ERROR: Matrix subtract format wrong";
        Matrix res(this->row(), this->col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = mat[i][j] - b.mat[i][j];
        return res;
    }

    Matrix operator*(const Matrix &b) const {
        if (this->col() != b.row())
            throw "ERROR: Matrix multiply format wrong";
        Matrix res(this->row(), b.col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                for (int k = 0; k < this->col(); k++)
                    res[i][j] += mat[i][k] * b.mat[k][j];
        return res;
    }

    Matrix operator*(const double &b) const {
        Matrix res;
        res.resize(this->row(), this->col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = mat[i][j] * b;
        return res;
    }

    // 矩阵点乘，重载^运算符
    Matrix operator^(const Matrix &b) const {
        if (this->row() != b.row() || this->col() != b.col())
            throw "ERROR: Matrix dot multiply format wrong";
        Matrix res;
        res.resize(this->row(), this->col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = mat[i][j] * b.mat[i][j];
        return res;
    }

    // 对标量除
    Matrix operator/(const double &b) const {
        Matrix res;
        res.resize(this->row(), this->col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = mat[i][j] / b;
        return res;
    }

    // 转置
    Matrix transpose() const {
        Matrix res;
        res.resize(this->col(), this->row());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = mat[j][i];
        return res;
    }

    Matrix toSigmoidMatrix() const {
        Matrix res(this->row(), this->col());
        for (int i = 0; i < res.row(); i++)
            for (int j = 0; j < res.col(); j++)
                res[i][j] = sigmoid(mat[i][j]);
        return res;
    }

    void print() {
        cout << row() << " * " << col() << endl;
        for (int i = 0; i < this->row(); i++) {
            for (int j = 0; j < this->col(); j++)
                cout << mat[i][j] << " ";
            cout << endl;
        }
    }

};

//打印图片
void printImage(Matrix &img) {
    for (int i = 0; i < IMAGE_ROW; i++) {
        for (int j = 0; j < IMAGE_COL; j++) {
            printf("%.2lf ", img[IMAGE_COL * i + j][0]);
        }
        cout << endl;
    }
}

//数据点
struct Point {
    Matrix image;
    Matrix label;

    Point(const char *image, uint8_t num) {
        this->image.resize(IMAGE_SIZE, 1);
        for (int i = 0; i < IMAGE_SIZE; i++) {
            this->image[i][0] = (uint8_t) image[i];
        }
        label.resize(OUT_SIZE, 1, 0);
        label[num][0] = 1;
    }
};

vector<Point> TrainData, TestData;

void readData(vector<Point> &train, vector<Point> &test) {
    char rubbish[16];
    ifstream train_images("./train-images.idx3-ubyte", ios::binary | ios::in);
    ifstream train_labels("./train-labels.idx1-ubyte", ios::binary | ios::in);
    if (train_images.fail()) {
        cout << "Train images not found";
        exit(0);
    }
    if (train_labels.fail()) {
        cout << "Train labels not found";
        exit(0);
    }
    train_images.read(rubbish, 16);
    train_labels.read(rubbish, 8);
    for (int i = 0; i < TRAIN_NUM; i++) {
        char image[IMAGE_SIZE];
        uint8_t num;
        train_images.read(image, IMAGE_SIZE);
        train_labels.read((char *) (&num), LABEL_SIZE);
        train.emplace_back(image, num);
    }

    ifstream test_images("./t10k-images.idx3-ubyte", ios::binary | ios::in);
    ifstream test_labels("./t10k-labels.idx1-ubyte", ios::binary | ios::in);
    if (test_images.fail()) {
        cout << "Test images not found";
        exit(0);
    }
    if (test_labels.fail()) {
        cout << "Test labels not found";
        exit(0);
    }
    test_images.read(rubbish, 16);
    test_labels.read(rubbish, 8);
    for (int i = 0; i < TEST_NUM; i++) {
        char image[IMAGE_SIZE];
        uint8_t num;
        test_images.read(image, IMAGE_SIZE);
        test_labels.read((char *) (&num), LABEL_SIZE);
        test.emplace_back(image, num);
    }
}

//归一化
void normalize(vector<Point> &set) {
    for (auto &p: set) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            p.image[i][0] /= 255.0;
        }
    }
}

//相关变量
vector<Matrix> Weight;     // 权重
vector<Matrix> Bias;       // 偏移量
vector<Matrix> Error;      // 误差
vector<Matrix> der_Weight; // 损失函数对权重偏导
vector<Matrix> der_Bias;   // 损失函数对偏差偏导
vector<Matrix> receive;    // 某层输入
vector<Matrix> activation; // 某层输出

void initialize() {
    srand(time(nullptr));

    // 初始化权重
    Weight.resize(LAYER_NUM);
    for (int i = 1; i < LAYER_NUM; i++) {
        Matrix &w = Weight[i];
        w.resize(NEURON_NUM[i], NEURON_NUM[i - 1]);
        for (int j = 0; j < w.row(); j++)
            for (int k = 0; k < w.col(); k++) {
                w[j][k] = ((double) (rand() % 1000) / 700 - 0.5) * sqrt(1.0 / NEURON_NUM[i - 1]);
            }
    }
    // 初始化
    receive.resize(LAYER_NUM);
    activation.resize(LAYER_NUM);
    Bias.resize(LAYER_NUM);
    Error.resize(LAYER_NUM);
    der_Weight.resize(LAYER_NUM);
    der_Bias.resize(LAYER_NUM);
    for (int i = 1; i < LAYER_NUM; i++) {
        receive[i].resize(NEURON_NUM[i], 1);
        activation[i].resize(NEURON_NUM[i], 1);
        Bias[i].resize(NEURON_NUM[i], 1);
        Error[i].resize(NEURON_NUM[i], 1);
        der_Weight[i].resize(NEURON_NUM[i], NEURON_NUM[i - 1]);
        der_Bias[i].resize(NEURON_NUM[i], 1);
    }
}

// 对某个样本前向传播
inline void forwardPropagation(const Point &point) {
    activation[0] = point.image;
    for (int i = 1; i < LAYER_NUM; i++) {
        receive[i] = Weight[i] * activation[i - 1];
        activation[i] = receive[i].toSigmoidMatrix();
    }
}

// 对某个样本反向传播
inline void backPropagation(const Point &point) {
    Error[LAYER_NUM - 1] = activation[LAYER_NUM - 1] - point.label;
    for (int i = LAYER_NUM - 2; i >= 1; i--) {
        Matrix ONE(activation[i].row(), activation[i].col(), 1);
        Error[i] = (Weight[i + 1].transpose() * Error[i + 1]) ^
                   (activation[i] ^ (ONE - activation[i]));
    }
}

// 将某个样本产生的偏导数累加
inline void accumulateDerivative() {
    for (int i = 1; i < LAYER_NUM; i++) {
        der_Weight[i] = der_Weight[i] + (Error[i] * activation[i - 1].transpose());
        der_Bias[i] = der_Bias[i] + Error[i];
    }
}

// 计算平均偏导
inline void calculateAverageDerivative() {
    for (int i = 1; i < LAYER_NUM; i++) {
        der_Weight[i] = der_Weight[i] / TRAIN_NUM;
        der_Bias[i] = der_Bias[i] / TRAIN_NUM;
    }
}

// 梯度下降
inline void gradientDescent() {
    for (int i = 1; i < LAYER_NUM; i++) {
        Weight[i] = Weight[i] - (der_Weight[i] * LEARNING_RATE);
        Bias[i] = Bias[i] - (der_Bias[i] * LEARNING_RATE);
    }
}

// 某个样本输出是否正确
inline bool match(const Matrix &res, const Matrix &label) {
    int max_pos = 0;
    for (int i = 1; i < OUT_SIZE; i++)
        if (res[i][0] > res[max_pos][0])
            max_pos = i;
    return label[max_pos][0] == 1;
}

// 训练集正确率
inline double evaluateStudy() {
    int cnt = 0;
    for (int i = 0; i < TRAIN_NUM; i++) {
        const Point &point = TrainData[i];
        forwardPropagation(point);
        if (match(activation[LAYER_NUM - 1], point.label))
            cnt++;
    }
    return (double) cnt / TRAIN_NUM;
}

// 测试集正确率
inline double evaluateTest() {
    int cnt = 0;
    for (int i = 0; i < TEST_NUM; i++) {
        const Point &point = TestData[i];
        forwardPropagation(point);
        if (match(activation[LAYER_NUM - 1], point.label))
            cnt++;
    }
    return (double) cnt / TEST_NUM;
}

// 输出参数
inline void showParameters() {
    cout << "权重: " << endl;
    for (int i = 1; i < LAYER_NUM; i++)
        Weight[i].print();
    cout << "偏移量: " << endl;
    for (int i = 1; i < LAYER_NUM; i++)
        Bias[i].print();
}

int main() {

    clock_t start_time = clock();

    readData(TrainData, TestData);
    normalize(TrainData);
    normalize(TestData);
    initialize();
    cout << "------ 训练参数 ------" << endl;
    cout << "训练集样本数: " << TRAIN_NUM << endl;
    cout << "训练集目标正确率: " << TARGET_TRAIN_CORRECTNESS << endl;
    cout << "迭代次数上限: " << ITERATION_CAP << endl;
    cout << "学习率: " << LEARNING_RATE << endl;
    cout << "---------------------" << endl;

    int t;
    for (t = 0; t < ITERATION_CAP && evaluateStudy() < TARGET_TRAIN_CORRECTNESS - EPS; t++) {
        try {
            for (int i = 0; i < LAYER_NUM; i++) {
                der_Weight[i].clear();
                der_Bias[i].clear();
            }
            for (int j = 0; j < TRAIN_NUM; j++) {
                forwardPropagation(TrainData[j]);
                backPropagation(TrainData[j]);
                accumulateDerivative();
            }
            calculateAverageDerivative();
            gradientDescent();
        } catch (char const *exception) {
            cout << exception << endl;
        }
    }

    cout << "------ 训练结果 ------" << endl;
    cout << "耗时: " << (double) (clock() - start_time) / CLOCKS_PER_SEC << "s" << endl;
    cout << "迭代次数: " << t << endl;
    cout << "训练集正确率: " << evaluateStudy() << endl;
    cout << "测试集样本数: " << TEST_NUM << endl;
    cout << "测试集正确率: " << evaluateTest() << endl;
    cout << "---------------------" << endl;

    system("pause");

    return 0;
}
