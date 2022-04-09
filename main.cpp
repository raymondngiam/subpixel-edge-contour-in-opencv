#include <iostream>
#include <fmt/core.h>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <execution>
#include <cmath>
#include <memory>
#include <boost/filesystem.hpp>
#include <numeric>
#include <set>

double ContourLengthSingle(const std::vector<cv::Point> &contour);
void ContourLength(std::vector<std::vector<cv::Point>> &contours, std::vector<double> &lengths);

std::shared_ptr<cv::Point2d> SubPixelFacet(const cv::Point& p,
                                           cv::Mat& gyMat,
                                           cv::Mat& gxMat,
                                           cv::Mat& gyyMat,
                                           cv::Mat& gxxMat,
                                           cv::Mat& gxyMat);

std::shared_ptr<std::vector<std::shared_ptr<cv::Point2d>>>
SubPixelSingle(cv::Mat &gy,
               cv::Mat &gx,
               cv::Mat &gyy,
               cv::Mat &gxx,
               cv::Mat &gxy,
               const std::vector<cv::Point> &cont);

void SubPixelEdgeContour(const cv::Mat &image_gray,
                         const std::vector<std::vector<cv::Point>> &filteredCont,
                         std::vector<std::shared_ptr<std::vector<std::shared_ptr<cv::Point2d>>>> &contSubPixFull);

void GetEdgeContourValidIndices(const std::vector<cv::Vec4i> &hierarchy, std::vector<int> &validIndices,
                            std::vector<int> &excludeIndices);

const std::vector<cv::Scalar_<double>> COLORS{
    cv::Scalar_<double>(0,0,255),
    cv::Scalar_<double>(0,255,0),
    cv::Scalar_<double>(255,0,0)
};

int main() {
    auto image_path = "../img";
    auto output_image_path = "./img_out";
    if(!boost::filesystem::exists(output_image_path)){
        boost::filesystem::create_directory(output_image_path);
    }
    auto filename = "calib_l_01.png";
    cv::Mat image;
    image = cv::imread(fmt::format("{}/{}",image_path,filename), cv::IMREAD_COLOR);
    cv::Mat image_gray;
    cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);

    fmt::print("{},{}\n",image.rows,image.cols);
    cv::Mat edgeIm = cv::Mat::zeros(image.rows,image.cols,CV_8UC1);
    cv::Canny(image_gray,edgeIm,180,200);
    cv::imwrite(fmt::format("{}/{}",output_image_path,"outputEdge.png"),edgeIm);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edgeIm,contours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_NONE);
    std::vector<int> validIndices;
    std::vector<int> excludeIndices;
    GetEdgeContourValidIndices(hierarchy, validIndices, excludeIndices);

    std::vector<std::vector<cv::Point>> innerContours;
    innerContours.resize(validIndices.size());
    std::transform(std::execution::par,
                   validIndices.begin(),
                   validIndices.end(),
                   innerContours.begin(),
                   [contours](int i){return contours[i];}
    );

    cv::Mat contourIm = cv::Mat::zeros(image.rows,image.cols,CV_8UC3);
    fmt::print("{}\n",innerContours.size());
    for (int i=0; i<innerContours.size(); i++){
        int thickness = 1;
        cv::drawContours(contourIm,innerContours,i,COLORS[i%COLORS.size()],thickness,cv::LINE_8);
    }
    cv::imwrite(fmt::format("{}/{}",output_image_path,"innerContours.png"),contourIm);

    std::vector<std::vector<cv::Point>> externalContours;
    externalContours.resize(excludeIndices.size());
    std::transform(std::execution::par,
                   excludeIndices.begin(),
                   excludeIndices.end(),
                   externalContours.begin(),
                   [contours](int i){return contours[i];}
    );

    cv::Mat contourExtIm = cv::Mat::zeros(image.rows,image.cols,CV_8UC3);
    fmt::print("{}\n",externalContours.size());
    for (int i=0; i<externalContours.size(); i++){
        int thickness = 1;
        cv::drawContours(contourExtIm,externalContours,i,COLORS[i%COLORS.size()],thickness,cv::LINE_8);
    }
    cv::imwrite(fmt::format("{}/{}",output_image_path,"externalContours.png"),contourExtIm);

    std::vector<double> contLengths;
    ContourLength(innerContours, contLengths);
    cv::Mat_<double> contLengthMat(contLengths);

    // extract properties
    std::vector<double> contRa, contAspectRatio;
    std::vector<cv::Point> contCenter;
    for(const auto& c : innerContours){
        auto M = cv::moments(c);

        double area = M.m00;
        auto centerX = int(M.m10/area);
        auto centerY = int(M.m01/area);
        double m20 = M.mu20/area;
        double m02 = M.mu02/area;
        double m11 = M.mu11/area;
        double c1 = m20-m02;
        double c2 = c1*c1;
        double c3 = 4*m11*m11;

        contCenter.emplace_back(centerX,centerY);

        auto ra = sqrt(2.0*(m20+m02+sqrt(c2+c3)));
        auto rb = sqrt(2.0*(m20+m02-sqrt(c2+c3)));
        contRa.emplace_back(ra);
        contAspectRatio.emplace_back(ra/rb);
    }

    cv::Mat_<double> contRadiusMat(contRa);
    cv::Mat_<double> contAspectRatioMat(contAspectRatio);

    fmt::print("{},{}\n",contRadiusMat.rows,contRadiusMat.cols);
    cv::Mat thresAspectRatio, thresRadius, thresContLength;
    const double RADIUS_MAX = 5;
    const double CONT_LENGTH_MAX = 2*M_PI*RADIUS_MAX;
    cv::threshold(contAspectRatioMat,thresAspectRatio,0.8,1.0,cv::ThresholdTypes::THRESH_BINARY);
    cv::threshold(contRadiusMat,thresRadius,RADIUS_MAX,1.0,cv::ThresholdTypes::THRESH_BINARY_INV);
    cv::threshold(contLengthMat,thresContLength,CONT_LENGTH_MAX,1.0,cv::ThresholdTypes::THRESH_BINARY_INV);

    cv::Mat and1, and2;
    cv::bitwise_and(thresAspectRatio,thresRadius,and1);
    cv::bitwise_and(and1,thresContLength,and2);
    fmt::print("Filtered object count: {}\n",std::accumulate(and2.begin<double>(),and2.end<double>(),0));

    cv::Mat filteredIdx;
    cv::findNonZero(and2,filteredIdx);

    std::vector<std::vector<cv::Point>> filteredCont;
    std::vector<cv::Point> filteredContCenter;
    for (int i=0; i<filteredIdx.rows; i++) {
        int index = filteredIdx.at<cv::Point>(i).y;
        filteredCont.emplace_back(innerContours[index]);
        filteredContCenter.emplace_back(contCenter[index].x,contCenter[index].y);
    }

    std::vector<std::shared_ptr<std::vector<std::shared_ptr<cv::Point2d>>>> contSubPixFull;
    SubPixelEdgeContour(image_gray, filteredCont, contSubPixFull);

    std::ofstream f("./data.txt",std::ios_base::out);
    f<<"contour_id,point_id,x,y"<<std::endl;
    for (int i = 0; i < contSubPixFull.size(); ++i) {
        for (int j=0; j<(*contSubPixFull[i]).size(); j++){
            auto p = *((*contSubPixFull[i])[j]);
            auto line = fmt::format("{0},{1},{2:.4f},{3:.4f}\n",
                                    i,
                                    j,
                                    p.x,
                                    p.y);
            f<<line;
        }
    }

    const int cropHalfWidth = 7;
    const int upScaleFactor = 50;
    const int finalResultCount = 4;

    for (int resultIndex = 0; resultIndex < finalResultCount; resultIndex++) {
        int xCropStart = filteredContCenter[resultIndex].x - cropHalfWidth;
        int yCropStart = filteredContCenter[resultIndex].y - cropHalfWidth;
        cv::Rect rect(xCropStart, yCropStart, 2 * cropHalfWidth + 1, 2 * cropHalfWidth + 1);

        cv::Mat crop = image(rect);

        int upScaledWidth = upScaleFactor * (2 * cropHalfWidth + 1);
        cv::Mat upScaled = cv::Mat::zeros(upScaledWidth, upScaledWidth, CV_8UC3);

        for (int i = 0; i < crop.cols; ++i) {
            for (int j = 0; j < crop.rows; ++j) {
                cv::Mat roi = upScaled(cv::Rect(i * upScaleFactor, j * upScaleFactor, upScaleFactor, upScaleFactor));
                roi = crop.at<cv::Vec<uchar, 3>>(j, i);
            }
        }

        std::vector<cv::Point> displayContour;
        for (const auto &p: *contSubPixFull[resultIndex]) {
            int x = floor(((p->x - xCropStart) + 0.5) * upScaleFactor);
            int y = floor(((p->y - yCropStart) + 0.5) * upScaleFactor);
            cv::drawMarker(upScaled, cv::Point(x, y), cv::Scalar(0.0, 255.0, 0.0), cv::MARKER_TILTED_CROSS, 20, 3);
            displayContour.emplace_back(x, y);
        }
        std::vector<std::vector<cv::Point>> displayContourFull{displayContour};
        cv::drawContours(upScaled, displayContourFull, 0, cv::Scalar(255.0, 0.0, 0.0), 3);

        cv::imwrite(fmt::format("{}/final-{:02d}.png", output_image_path, resultIndex), upScaled);
    }
    return 0;
}

// For non maximum surpressed edge images, contour lines are single pixel in width.
// For closed contours, there are two possible outcomes from the boundary tracing algorithm,
// namely inner (hole), or external (non-hole) contour.
// OpenCV `findContours` with `RETR_CCOMP` option returns hierarchy list that starts with an external contour.
// Iterate through all external contours in the hierarchy list by following the `NEXT_SAME` indices;
// if the current external contour does have a child, this indicates that it is a false positive that
// corresponds to another inner hole contour in the set. Thus, we add it into the `excludeIndices` list.
void GetEdgeContourValidIndices(const std::vector<cv::Vec4i> &hierarchy, std::vector<int> &validIndices,
                                   std::vector<int> &excludeIndices) {
    const int NEXT_SAME = 0;
    const int PREV_SAME = 1;
    const int FIRST_CHILD = 2;
    const int PARENT = 3;

    int index=0;
    while (index != -1){
        if (hierarchy[index][FIRST_CHILD]!=-1){
            excludeIndices.emplace_back(index);
        }
        index = hierarchy[index][NEXT_SAME];
    }

    std::vector<int> l(hierarchy.size());
    std::iota(l.begin(),l.end(),0);
    std::set<int> setFullIndices(l.begin(),l.end());
    std::set_difference(setFullIndices.begin(),
                        setFullIndices.end(),
                        excludeIndices.begin(),
                        excludeIndices.end(),
                        std::back_inserter(validIndices)
                        );
}

void SubPixelEdgeContour(const cv::Mat &image_gray,
                         const std::vector<std::vector<cv::Point>> &filteredCont,
                         std::vector<std::shared_ptr<std::vector<std::shared_ptr<cv::Point2d>>>> &contSubPixFull) {
    // 7-tap interpolant and 1st and 2nd derivative coefficients according to
    // H. Farid and E. Simoncelli, "Differentiation of Discrete Multi-Dimensional Signals"
    // IEEE Trans. Image Processing. 13(4): pp. 496-508 (2004)
    std::vector<double> p_vec{0.004711,  0.069321,  0.245410,  0.361117,  0.245410,  0.069321,  0.004711};
    std::vector<double> d1_vec{-0.018708,  -0.125376,  -0.193091,  0.000000, 0.193091, 0.125376, 0.018708};
    std::vector<double> d2_vec{0.055336,  0.137778, -0.056554, -0.273118, -0.056554,  0.137778,  0.055336};

    auto p = cv::Mat_<double>(p_vec);
    auto d1 = cv::Mat_<double>(d1_vec);
    auto d2 = cv::Mat_<double>(d2_vec);

    cv::Mat dx, dy, grad;
    cv::sepFilter2D(image_gray,dy,CV_64F,p,d1);
    cv::sepFilter2D(image_gray,dx,CV_64F,d1,p);
    cv::pow(dy.mul(dy,1.0) + dx.mul(dx,1.0),0.5,grad);

    cv::Mat gy, gx, gyy, gxx, gxy;
    cv::sepFilter2D(grad,gy,CV_64F,p,d1);
    cv::sepFilter2D(grad,gx,CV_64F,d1,p);
    cv::sepFilter2D(grad,gyy,CV_64F,p,d2);
    cv::sepFilter2D(grad,gxx,CV_64F,d2,p);
    cv::sepFilter2D(grad,gxy,CV_64F,d1,d1);

    contSubPixFull.resize(filteredCont.size());
    std::transform(std::execution::par,
                   filteredCont.cbegin(),
                   filteredCont.cend(),
                   contSubPixFull.begin(),
                   [&gy,&gx,&gyy,&gxx,&gxy](const std::vector<cv::Point>& cont){
        return SubPixelSingle(gy,gx,gyy,gxx,gxy,cont);}
                   );
}

std::shared_ptr<std::vector<std::shared_ptr<cv::Point2d>>>
SubPixelSingle(cv::Mat &gy,
               cv::Mat &gx,
               cv::Mat &gyy,
               cv::Mat &gxx,
               cv::Mat &gxy,
               const std::vector<cv::Point> &cont) {
    auto contSubPix = std::make_shared<std::vector<std::shared_ptr<cv::Point2d>>>();
    contSubPix->resize(cont.size());
    std::transform(std::execution::seq,
                   cont.cbegin(),
                   cont.cend(),
                   contSubPix->begin(),
                   [&gy,&gx,&gyy,&gxx,&gxy](const cv::Point& p){ return SubPixelFacet(p,gy,gx,gyy,gxx,gxy); }
                   );
    return contSubPix;
}

// Subpixel edge extraction method according to
// C. Steger, "An unbiased detector of curvilinear structures",
// IEEE Transactions on Pattern Analysis and Machine Intelligence,
// 20(2): pp. 113-125, (1998)
std::shared_ptr<cv::Point2d> SubPixelFacet(const cv::Point& p,
                                           cv::Mat& gyMat,
                                           cv::Mat& gxMat,
                                           cv::Mat& gyyMat,
                                           cv::Mat& gxxMat,
                                           cv::Mat& gxyMat){
    auto row = p.y;
    auto col = p.x;
    auto gy = gyMat.at<double>(row,col);
    auto gx = gxMat.at<double>(row,col);
    auto gyy = gyyMat.at<double>(row,col);
    auto gxx = gxxMat.at<double>(row,col);
    auto gxy = gxyMat.at<double>(row,col);

    Eigen::Matrix<double,2,2>hessian;
    hessian << gyy,gxy,gxy,gxx;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(hessian, Eigen::ComputeFullV);
    auto v = svd.matrixV();
    // first column vector of v, corresponding to largest eigen value
    // is the direction perpendicular to the line
    auto ny = v(0,0);
    auto nx = v(1,0);
    auto t=-(gx*nx + gy*ny)/(gxx*nx*nx + 2*gxy*nx*ny + gyy*ny*ny);
    auto px=t*nx;
    auto py=t*ny;

    return std::make_shared<cv::Point2d>(col+px,row+py);
}

void ContourLength(std::vector<std::vector<cv::Point>> &contours, std::vector<double> &lengths) {
    lengths.resize(std::distance(contours.begin(), contours.end()));
    std::transform(std::execution::par,
                   contours.begin(),
                   contours.end(),
                   lengths.begin(),
                   ContourLengthSingle
                   );
}

// contour length
double ContourLengthSingle(const std::vector<cv::Point> &contour) {
    std::vector<double> lengths;
    for (int i =1; i<contour.size();i++){
        double distx = contour[i].x - contour[i-1].x;
        double disty = contour[i].y - contour[i-1].y;
        double dist = std::sqrt(distx*distx + disty*disty);
        lengths.emplace_back(dist);
    }
    double length = std::accumulate(lengths.begin(),lengths.end(),0,[](double a, double b){ return a+b; });
    return length;
}