#include <fcl/fcl.h>
#include <vector>
#include <memory>
using Model=fcl::BVHModel<fcl::OBBRSSd>;
extern "C" void* create(const double* verts, int nv, const int* tri, int nt){
 auto m=new Model();
 std::vector<fcl::Vector3d> v;
 std::vector<fcl::Triangle> t;
 for(int i=0;i<nv;i++) v.emplace_back(verts[3*i],verts[3*i+1],verts[3*i+2]);
 for(int i=0;i<nt;i++) t.emplace_back(tri[3*i],tri[3*i+1],tri[3*i+2]);
 m->beginModel();m->addSubModel(v,t);m->endModel();return m;
}
extern "C" double distance_at(void* handle, const double* xyz, const double* xyzw){
 fcl::Boxd cube(.06,.06,.06);
 fcl::Transform3d pose=fcl::Transform3d::Identity();
 pose.linear()=Eigen::Quaterniond(xyzw[3],xyzw[0],xyzw[1],xyzw[2]).normalized().toRotationMatrix();
 pose.translation()=Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
 fcl::DistanceRequestd request; fcl::DistanceResultd result;
 double d=fcl::distance(static_cast<Model*>(handle),fcl::Transform3d::Identity(),&cube,pose,request,result);
 return std::max(d,0.);
}
extern "C" void distances(void* model, int count, const double* xyz, const double* xyzw, float* result) {
    for(int i=0;i<count;++i) {
        result[i]=1000.0*distance_at(model,xyz+3*i,xyzw+4*i);
    }
}
