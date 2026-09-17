#pragma once
/**********************************************************************
 *  shape_filter.hpp  ―  라바콘·드럼통 판별용 기하학·RANSAC 필터
 *  - isVerticalSlender : 전봇대/가로수(세로로 긴 물체) 거르기
 *  - isLongCurb        : 연석(길이≫높이) 거르기
 *  - isCylinderOrCone  : RANSAC 기반 원통/원뿔 적합
 *
 *  □ 의존성 최소화
 *    · PCL 1.8/1.10 공통 헤더만 사용
 *    · 노멀 추정·FromNormals API 제거 → 헤더 부족 환경에서도 빌드 가능
 *********************************************************************/

#include "clustering.hpp"

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace shape_filter
{

/* ────────────────────────────────────────────────────────── */
/*  세로로 가늘고 긴 물체(폴대·가로수) 판별 λ₁/λ₂ > ratio_thr */
inline bool isVerticalSlender(const pcl::PointCloud<pcl::PointXYZI>& c,
                              double ratio_thr)
{
    auto eval = clustering::eigenValues(c);       // λ₁ ≥ λ₂ ≥ λ₃
    return (eval.x() / eval.y()) > ratio_thr;
}

/*  길이 ≫ 높이인 연석·벽체 판별 λ₂/λ₃ > len_ratio            */
inline bool isLongCurb(const pcl::PointCloud<pcl::PointXYZI>& c,
                       double len_ratio)
{
    auto eval = clustering::eigenValues(c);
    return (eval.y() / eval.z()) > len_ratio;
}

/* ────────────────────────────────────────────────────────── */
/*  RANSAC 원통/원뿔 적합                                    */
inline bool isCylinderOrCone(const pcl::PointCloud<pcl::PointXYZI>& c,
                             double dist_thr_cyl,
                             double dist_thr_cone,
                             double inlier_ratio)
{
    /* 1) 점 수 체크 */
    if (c.size() < 30)
        return false;

    /* 공통 세그먼트 객체 */
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ModelCoefficients coeff;
    pcl::PointIndices inliers;

    auto runRansac = [&](auto model_type, double dist_thresh) -> bool
    {
        seg.setOptimizeCoefficients(true);
        seg.setModelType(model_type);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(dist_thresh);
        seg.setMaxIterations(100);
        seg.setInputCloud(c.makeShared());

        seg.segment(inliers, coeff);
        if (inliers.indices.empty()) return false;

        double ratio = static_cast<double>(inliers.indices.size()) / c.size();
        return ratio >= inlier_ratio;
    };

    /* 2) Cylinder 우선 시도 */
    if (runRansac(pcl::SACMODEL_CYLINDER, dist_thr_cyl))
        return true;

    /* 3) Cone 재시도 */
    if (runRansac(pcl::SACMODEL_CONE, dist_thr_cone))
        return true;

    return false;   // 둘 다 실패 → 라바콘/드럼통 아님
}

} // namespace shape_filter
