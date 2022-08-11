// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"
#include "pcl-1.8/pcl/filters/impl/filter.hpp"

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<RsPointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<RsPointXYZIRT>::Ptr laserCloudInRing;
    pcl::PointCloud<RsPointXYZIRT>::Ptr DAE_Cloud; // "Depth Azimuth Elevation" instead of "x y z"
    pcl::PointCloud<RsPointXYZIRT>::Ptr downsampled_DAE_Cloud;
    pcl::PointCloud<RsPointXYZIRT>::Ptr downsampled_DAE_Cloud2; // for debug
    pcl::PointCloud<RsPointXYZIRT>::Ptr filteredCloud; // for debug
    pcl::PointCloud<RsPointXYZIRT>::Ptr fullCloud;    
    pcl::PointCloud<RsPointXYZIRT>::Ptr fullInfoCloud; 
    pcl::PointCloud<RsPointXYZIRT>::Ptr groundCloud;
    pcl::PointCloud<RsPointXYZIRT>::Ptr segmentedCloud;
    pcl::PointCloud<RsPointXYZIRT>::Ptr segmentedCloudPure;
    pcl::PointCloud<RsPointXYZIRT>::Ptr outlierCloud;

    RsPointXYZIRT nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

    std::vector<int> point_idx;
    std::vector<int> point_row;
    std::vector<int> point_col;
    bool point_filtered = false;

public:
    ImageProjection():
        nh("~"){

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;
        nanPoint.ring = std::numeric_limits<float>::quiet_NaN();
        nanPoint.timestamp = std::numeric_limits<float>::quiet_NaN();

        allocateMemory();
        resetParameters();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<RsPointXYZIRT>());
        laserCloudInRing.reset(new pcl::PointCloud<RsPointXYZIRT>());
        DAE_Cloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        downsampled_DAE_Cloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        downsampled_DAE_Cloud2.reset(new pcl::PointCloud<RsPointXYZIRT>());
        filteredCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        fullInfoCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());

        groundCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        segmentedCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());
        segmentedCloudPure.reset(new pcl::PointCloud<RsPointXYZIRT>());
        outlierCloud.reset(new pcl::PointCloud<RsPointXYZIRT>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        segMsg.endOrientation = 0.0;
        segMsg.startOrientation = 0.0;

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        // decode Header part
        /*
            sensor_msgs::PointCloud2.header
            {
                uint32 seq; // # sequence ID: consecutively increasing ID 
                time stamp; // includes stamp.sec(seconds) since epoch, and stamp.nsec(nanoseconds) since stamp.secs
                string frame_id; // #Frame this data is associated with
            }
        */
        cloudHeader = laserCloudMsg->header; 
        // cout<<"-----------------------------"<<endl;
        // cout<<cloudHeader.seq<<endl;
        // cout<<cloudHeader.stamp.sec+round(cloudHeader.stamp.nsec/1e7)/1e2 <<endl;
        // cout<<cloudHeader.frame_id<<endl;


        // decode PointCloud part
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        
        // Robosense lidar's ring does not represent row number!
        // have "ring" channel in the cloud
        // if (useCloudRing == true){
        //     pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
        //     pcl::removeNaNFromPointCloud(*laserCloudInRing, *laserCloudInRing, indices);
        // }
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // PrintPointByRes(); // for debug

        // 2. filter points
        FilterPointCloud();
        // 3. Range image projection
        // projectPointCloud() in step2

        // 4. Mark ground points
        groundRemoval();

        // 5. Point cloud segmentation
        cloudSegmentation();

        // 6. Publish all clouds
        publishCloud();

        // 7. Reset parameters for next iteration
        resetParameters();

    }

    void PrintPointByRes()
    {
        // print point cloud by resolution of RoboSense-M1 
        // (elevation and azimuth are rounded to 0.1 degree)
        cout<<endl<<"------------------------------PrintPointByRes------------------------------"<<endl;

        std::map<double, int> elevation_pointNum; 
        std::map<double, int> azimuth_pointNum;
        for (int i=0;i<laserCloudIn->points.size();i++)
        {
            RsPointXYZIRT point;
            point = laserCloudIn->points[i];
            double randius = sqrt(pow(point.x,2) +pow(point.y,2) +pow(point.z,2) );
            double ele_degree = asin(point.z/randius) * 180 /M_PI; //[-Pi/2, Pi/2]
            ele_degree = double(round(ele_degree*100))/100 ;
            // if ((ele_degree>=-12.5)&&(ele_degree<=12.5))
            // {
                auto ret_found1 = elevation_pointNum.find(ele_degree);
                if (ret_found1 == elevation_pointNum.end())
                    elevation_pointNum.insert({ele_degree, 1});
                else
                    elevation_pointNum.at(ele_degree)++;
            // }
                

            double azi_degree = atan2(point.y, point.x) * 180 /M_PI;  // reverse y-axis orientation, negtive values for left side, positive for right side.
            azi_degree = double(round(azi_degree*100))/100;
            // if ((azi_degree>=-60.0)&&(azi_degree<=60.0))
            // {
                auto ret_found2 = azimuth_pointNum.find(azi_degree);
                if (ret_found2 == azimuth_pointNum.end())
                    azimuth_pointNum.insert({azi_degree, 1});
                else
                    azimuth_pointNum.at(azi_degree)++;
                
            // }
        }
        
        int count_point=0;
        float min_val=0.0, max_val=0.0;
        for(auto i:elevation_pointNum)
        {
            count_point += i.second;
            if (i.first<min_val)
                min_val = i.first;
            if (i.first>max_val)
                max_val = i.first;
            // cout<<i.first<<":"<<i.second<<", ";
        }
        cout<<"SUM elevation: "<<count_point<<", min_ele: "<<min_val<<", max_ele: "<< max_val<<endl<<endl;


        count_point=0;
        min_val=0.0, max_val=0.0;
        for(auto i:azimuth_pointNum)
        {
            count_point += i.second;
            if (i.first<min_val)
                min_val = i.first;
            if (i.first>max_val)
                max_val = i.first;
            // cout<<i.first<<":"<<i.second<<", ";
        }
        cout<<"SUM azimuth: "<<count_point<<", min_azi: "<<min_val<<", max_azi: "<< max_val<<endl<<endl;
    

        // print min/max x/y value in point cloud.

        float miny=0, maxy=0;
        float minx=0, maxx=0;
        for (int i=0;i<laserCloudIn->points.size();i++)
        {
            if (laserCloudIn->points[i].y>maxy)
                maxy = laserCloudIn->points[i].y;
            if (laserCloudIn->points[i].y<miny)
                miny = laserCloudIn->points[i].y;
            if (laserCloudIn->points[i].x>maxx)
                maxx = laserCloudIn->points[i].x;
            if (laserCloudIn->points[i].x<minx)
                minx = laserCloudIn->points[i].x;
        }
        cout<<"MinY->MaxY: "<<miny<<","<<maxy<<endl;
        cout<<"MinX->MaxX: "<<minx<<","<<maxx<<endl;    
    }

    // to filter the points outside of threshold-region.
    bool is_point_valid(float ele_value, float azi_value)
    {
        if (abs(ele_value) > elevation_thres)
            return false;
        else if (abs(azi_value) > azimuth_thres)
            return false;
        else
            return true;
    }
    
    void get_minmax_val(float val, float& min, float& max)
    {
        if (val<min)
            min = val;
        if (val>max)
            max = val;
    }
    
    // fill map_ele_azi_idx
    void construct_point_mat(float ele_val, float azi_val, int point_index, std::map<float,std::map<float, int> >& map_ele_azi_idx)
    {
        int ele_index = round(ele_val/ang_res_y); // bottom to top, [-90, 90]/ang_res_y
        int azi_index = round(azi_val/ang_res_x); // right to left, [-90, 90]/ang_res_y
        float new_dist = pow(DAE_Cloud->points[point_index].y - azi_index*ang_res_x, 2) + 
                                pow(DAE_Cloud->points[point_index].z - ele_index*ang_res_y, 2);
        
        if (sqrt(new_dist) > min(ang_res_x, ang_res_y)/2)
        {
            cout<<"pass."<<endl;
            return;
        }
            
        auto ele_found1 = map_ele_azi_idx.find(ele_index);
        if (ele_found1 == map_ele_azi_idx.end()) // not found
        {
            std::map<float, int> map_azi_idx;
            map_azi_idx.insert({azi_index, point_index});
            map_ele_azi_idx.insert({ele_index, map_azi_idx});
        }
        else
        {
            auto azi_found1 = map_ele_azi_idx.at(ele_index).find(azi_index);
            if (azi_found1 == map_ele_azi_idx.at(ele_index).end())
                map_ele_azi_idx.at(ele_index).insert({azi_index, point_index});
            else
            {                
                int curr_index = map_ele_azi_idx.at(ele_index).at(azi_index);

                // // by intensity
                // float curr_intensity = DAE_Cloud->points[curr_index].intensity;
                // if (DAE_Cloud->points[point_index].intensity > curr_intensity)
                //     map_ele_azi_idx.at(ele_index).at(azi_index) = point_index;

                // by dist 
                float curr_dist = pow(DAE_Cloud->points[curr_index].y - azi_index*ang_res_x, 2) + 
                                pow(DAE_Cloud->points[curr_index].z - ele_index*ang_res_y, 2);                
                if (new_dist < curr_dist)
                    map_ele_azi_idx.at(ele_index).at(azi_index) = point_index;

            }
        }
    }

    // debug
    void get_filteredCloud()
    {
        filteredCloud->points.resize(point_idx.size());
        for (int i=0; i<point_idx.size(); i++)
        {
            filteredCloud->points[i] = laserCloudIn->points[point_idx[i]];
        }
    }
    
    // debug
    void fill_downsampled_DAE_Cloud(std::map<float,std::map<float, int> >& map_ele_azi_idx)
    {
        // generate downsampled_point_num for debug
        int downsampled_point_num = 0;
        for (const auto& ele_azi_idx : map_ele_azi_idx)
           downsampled_point_num += ele_azi_idx.second.size();

        cout<<"downsampled_DAE_Cloud includes "<<downsampled_point_num<<" points."<<endl;

        downsampled_DAE_Cloud->points.resize(downsampled_point_num);
        std::fill(downsampled_DAE_Cloud->points.begin(), downsampled_DAE_Cloud->points.end(), nanPoint);
        downsampled_DAE_Cloud2->points.resize(downsampled_point_num);
        std::fill(downsampled_DAE_Cloud2->points.begin(), downsampled_DAE_Cloud2->points.end(), nanPoint);
        int d_idx = 0;
        int minrow=0, maxrow=0, mincol=0, maxcol=0;
        for (const auto& ele_azi_idx : map_ele_azi_idx)
        {
            if (maxrow<ele_azi_idx.first)
                maxrow = ele_azi_idx.first;
            if (minrow>ele_azi_idx.first)
                minrow = ele_azi_idx.first;
            for (const auto& azi_idx : ele_azi_idx.second)
            {
                if (maxcol<azi_idx.first)
                    maxcol = azi_idx.first;
                if (mincol>azi_idx.first)
                    mincol = azi_idx.first;

                downsampled_DAE_Cloud->points[d_idx].x = DAE_Cloud->points[azi_idx.second].x;
                downsampled_DAE_Cloud->points[d_idx].y = DAE_Cloud->points[azi_idx.second].y;
                downsampled_DAE_Cloud->points[d_idx].z = DAE_Cloud->points[azi_idx.second].z;
                downsampled_DAE_Cloud->points[d_idx].intensity = DAE_Cloud->points[azi_idx.second].intensity;
                downsampled_DAE_Cloud->points[d_idx].ring      = DAE_Cloud->points[azi_idx.second].ring;
                downsampled_DAE_Cloud->points[d_idx].timestamp = DAE_Cloud->points[azi_idx.second].timestamp;

                // map_ele_azi_idx[ele_azi_idx.first][azi_idx.first] = d_idx; // now, map_ele_azi_idx saved point-index in downsampled_DAE_Cloud, instead of that in DAE_Cloud.
                d_idx++;
            }
        }
        cout<<"row: ["<<minrow<<", "<<maxrow<<"]; col: ["<<mincol<<", "<<maxcol<<"]"<<endl;
    }

    void projectPointCloud()
    {
        // range image projection
        // construct rangeMat to save range value for every point.
        // construct fullCloud to save XYZ(I1)RT for points. I1 stands for a float value, which means ROW.COL .
        // construct fullInfoCloud to save XYZ(I2)RT for points. I2 stands for the point's range value.



        uint rowIdn, columnIdn, index;

        RsPointXYZIRT thisPoint;
        for (int i=0; i<point_idx.size(); i++)
        {
            thisPoint.x = laserCloudIn->points[point_idx[i]].x;
            thisPoint.y = laserCloudIn->points[point_idx[i]].y;
            thisPoint.z = laserCloudIn->points[point_idx[i]].z;
            thisPoint.ring = laserCloudIn->points[point_idx[i]].ring;
            thisPoint.timestamp = laserCloudIn->points[point_idx[i]].timestamp;

            float range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;

            thisPoint.intensity = (float)point_row[i] + (float)point_col[i] / 10000.0; //row.col

            index = point_col[i]  + point_row[i] * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"

            rangeMat.at<float>(point_row[i], point_col[i]) = range;
        }
    }

    // filter laserCloudIn, then save in fullCloud.
    // elevation: [-12.5, 12.5]
    // azimuth: [-60.0, 60.0]
    void FilterPointCloud()
    {
        if (point_filtered == false)
        {
            int num_point = laserCloudIn->points.size();
            DAE_Cloud->points.resize(num_point);
            std::fill(DAE_Cloud->points.begin(), DAE_Cloud->points.end(), nanPoint);

            if (num_point > 0)
            {
                segMsg.startOrientation = laserCloudIn->points[0].timestamp;
                segMsg.endOrientation = laserCloudIn->points[0].timestamp;
            }

            // float max_ele=0.0, min_ele=0.0;
            // float max_azi=0.0, min_azi=0.0;
            float depth=0.0, azimuth=0.0, elevation=0.0;
            std::map<float, std::map<float, int> > map_ele_azi_idx;
            for (int i=0; i<num_point;i++)
            {
                RsPointXYZIRT curr_point = laserCloudIn->points[i];
                float range = sqrt(pow(curr_point.x,2) +pow(curr_point.y,2) +pow(curr_point.z,2) );
                elevation = asin(curr_point.z/range) * 180 /M_PI; //[-90, 90]
                azimuth = atan2(curr_point.y, curr_point.x) * 180 /M_PI; //[-90, 90]
                depth = curr_point.x;
                if (is_point_valid(elevation, azimuth) == false)
                    continue;

                DAE_Cloud->points[i].x = 10;
                DAE_Cloud->points[i].y = azimuth;
                DAE_Cloud->points[i].z = elevation;
                DAE_Cloud->points[i].intensity = curr_point.intensity;
                DAE_Cloud->points[i].ring      = curr_point.ring;
                DAE_Cloud->points[i].timestamp = curr_point.timestamp;

                construct_point_mat(elevation, azimuth, i, map_ele_azi_idx);

                if (curr_point.timestamp < segMsg.startOrientation)
                    segMsg.startOrientation = curr_point.timestamp;
                if (curr_point.timestamp > segMsg.endOrientation)
                    segMsg.endOrientation = curr_point.timestamp;
            }
            segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;

            // for debug
            // fill_downsampled_DAE_Cloud(map_ele_azi_idx);
            uint rowIdn, columnIdn;
            for (const auto& ele_azi_idx : map_ele_azi_idx)
            {
                for (const auto& azi_idx : ele_azi_idx.second)
                {
                    rowIdn = ele_azi_idx.first + round(elevation_thres/ang_res_y); // e.g. [-6, 6] --> [0, 12]
                    if ((rowIdn<2)||(rowIdn>21))
                        continue;

                    columnIdn = azi_idx.first + round(azimuth_thres/ang_res_x); // e.g. [-300, 300] --> [0, 600]
                    if ((columnIdn<150)|| (columnIdn>550))
                        continue;

                    point_idx.push_back(azi_idx.second);
                    point_row.push_back(rowIdn);
                    point_col.push_back(columnIdn);
                }
            }

            point_filtered = true;            
        }

        cout<< "point_idx length: "<< point_idx.size()<< endl;

        // get_filteredCloud();

        projectPointCloud();
    }

    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {

            // drop 4 points at begining of every row(scan)
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            // drop last 5 points of every row(scan)
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        // constrct labelMat:
        //          0:  init value, before segmentation;
        //         -1:  ground and invalid point;
        //          N:  Class N, after segmentation;
        //     999999:  invalid segmentation;

        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);

        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        

        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0){
            // pcl::toROSMsg(*DAE_Cloud, laserCloudTemp);
            // pcl::toROSMsg(*downsampled_DAE_Cloud, laserCloudTemp);
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }

        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            // pcl::toROSMsg(*filteredCloud, laserCloudTemp);
            // pcl::toROSMsg(*downsampled_DAE_Cloud2, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            // pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            pcl::toROSMsg(*laserCloudIn, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
