#include <iostream>
#include <map>
#include <iomanip>
#include <cmath>
#include <fstream>
#include "EcoMug.h"



EMLog::TLogLevel EMLog::ReportingLevel = WARNING;

double plane_width = 24.4;
double plane_height = 7.77;
double plane_separation = 2.44;

std::vector<std::vector<double>> getIntersections (std::array<double, 3>& x0, std::array<double, 3>& dir, bool & hit1, bool & hit2) {
    double t1 = (-plane_separation/2 - x0[0])/dir[0];
    double t2 = (plane_separation/2 - x0[0])/dir[0];
    std::vector<double> int_1 = {x0[0] + t1*dir[0], x0[1] + t1*dir[1], x0[2] + t1*dir[2]};
    std::vector<double> int_2 = {x0[0] + t2*dir[0], x0[1] + t2*dir[1], x0[2] + t2*dir[2]};
    hit1 = (int_1[2] > 0) && (int_1[2] < plane_height) && (int_1[1] > -plane_width/2) && (int_1[1] < plane_width/2);
    hit2 = (int_2[2] > 0) && (int_2[2] < plane_height) && (int_2[1] > -plane_width/2) && (int_2[1] < plane_width/2);
    hit1 &= t1 > 0;
    hit2 &= t2 > 0;
    return {{t1,t2},int_1,int_2};
}

int main(int argc, char **argv) {

    EcoMug muonGen;

    muonGen.SetUseCylinder();
    muonGen.SetCylinderRadius(atof(argv[1])*EMUnits::m);
    muonGen.SetCylinderHeight(atof(argv[2])*EMUnits::m);
    muonGen.SetCylinderCenterPosition({0., 0., (atof(argv[2])/2)*EMUnits::m});

    // muonGen.SetUseHSphere(); // half-spherical surface generation
    // muonGen.SetHSphereRadius(atof(argv[1])*EMUnits::m); // half-sphere radius
    // // (x,y,z) position of the center of the half-sphere
    // muonGen.SetHSphereCenterPosition({{0., 0., (atof(argv[1])/2)*EMUnits::m}});

    int N = atof(argv[argc-2]);
    double rate =  N/muonGen.GetEstimatedTime(N);
    std::cout << "Estimated rate [Hz] = " << rate << std::endl;

    // The array storing muon generation position
    std::array<double, 3> muon_position;
    std::array<double, 3> muon_dir;
    bool hit1;
    bool hit2;

    int hit1_count = 0, hit2_count =0, hitboth_count = 0;

    std::vector<std::vector<double>> time_and_intersections;

    std::ofstream out(argv[argc-1]);
    out << "hit1 hit2 hitboth t1 t2 int1_x int1_y int1_z int2_x int2_y int2_z theta phi x0 y0 z0 dx dy dz" << std::endl;


    for (auto i = 0; i < N; ++i) {
        if (i%(N/100)==0) std::cout << i << " out of " << N << std::endl;
        muonGen.Generate();
        muon_position = muonGen.GetGenerationPosition();
        double muon_theta = muonGen.GetGenerationTheta();
        double muon_phi = muonGen.GetGenerationPhi();
        muon_dir = {sin(muon_theta)*cos(muon_phi),sin(muon_theta)*sin(muon_phi),cos(muon_theta)};
        time_and_intersections = getIntersections(muon_position, muon_dir, hit1,hit2);
        if (hit1 && !hit2) ++hit1_count;
        else if (!hit1 && hit2) ++hit2_count;
        else if (hit1 && hit2) ++hitboth_count;
        if (!hit1 && !hit2) continue;
        out << (hit1 && !hit2) << " " << (!hit1 && hit2) << " " << (hit1 && hit2) << " ";
        out << time_and_intersections[0][0] << " " << time_and_intersections[0][1] << " ";
        out << time_and_intersections[1][0] << " " << time_and_intersections[1][1] << " " << time_and_intersections[1][2] << " ";
        out << time_and_intersections[2][0] << " " << time_and_intersections[2][1] << " " << time_and_intersections[2][2] << " ";
        out << muon_theta << " " << muon_phi << " ";
        out << muon_position[0] << " " << muon_position[1] << " " << muon_position[2] << " ";
        out << muon_dir[0] << " " << muon_dir[1] << " " << muon_dir[2] << std::endl;
    }
    out.close();

    std::cout << "Plane 1 was hit " << hit1_count << " out of " << N << " times" << std::endl;
    std::cout << "Plane 2 was hit " << hit2_count << " out of " << N << " times" << std::endl;
    std::cout << "Both planes were hit " << hitboth_count << " out of " << N << " times" << std::endl;
    std::cout << "Coincidence rate: " << rate * double(hitboth_count)/N << std::endl;

}