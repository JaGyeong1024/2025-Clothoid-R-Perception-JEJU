#pragma once
namespace ls_param
{
    constexpr double CLUSTER_DIST      = 0.18;   // [m]
    constexpr int    MIN_CLUSTER_PTS   = 8;

    constexpr double CYL_DIST_THRESH   = 0.03;   // [m]
    constexpr double CONE_DIST_THRESH  = 0.03;   // [m]
    constexpr double INLIER_RATIO      = 0.70;

    constexpr double VERTICAL_RATIO    = 10.0;   // λ₁/λ₂ (slender pole)
    constexpr double CURB_LEN_RATIO    = 4.0;    // length/height (curb)

    constexpr double Z_STD_MAX         = 0.15;   // [m]

    constexpr double INTENSITY_LOW     = 60.0;
    constexpr double INTENSITY_HIGH    = 150.0;

    constexpr double Z_MIN             = 0.10;   // [m]
    constexpr double Z_MAX             = 1.20;   // [m]
}
