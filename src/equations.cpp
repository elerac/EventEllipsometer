#include <utility>
#include <Eigen/Core>

// Calc numerator and denominator of the rational function
std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>, Eigen::Matrix<float, Eigen::Dynamic, 16>>
calcNumenatorDenominatorCoffs(const Eigen::VectorXf &thetaVector, float phi1, float phi2)
{
    size_t size = thetaVector.size();
    Eigen::Matrix<float, Eigen::Dynamic, 16> numenator_coffs(size, 16);
    Eigen::Matrix<float, Eigen::Dynamic, 16> denominator_coffs(size, 16);

    for (size_t i = 0; i < size; i++)
    {
        float theta = thetaVector(i);

        float sin2mulphi1add2multheta = sin(2 * phi1 + 2 * theta);
        float cos2mulphi1add2multheta = cos(2 * phi1 + 2 * theta);
        float sin2mulphi2add10multheta = sin(2 * phi2 + 10 * theta);
        float cos2mulphi2add10multheta = cos(2 * phi2 + 10 * theta);

        float tmp0 = cos2mulphi1add2multheta * sin2mulphi1add2multheta;
        float tmp1 = 4.0 * tmp0;
        float tmp2 = cos2mulphi1add2multheta * cos2mulphi1add2multheta;
        float tmp3 = 2.0 * tmp2;
        float tmp4 = sin2mulphi1add2multheta * sin2mulphi1add2multheta;
        float tmp5 = 2.0 * tmp4;
        float tmp6 = 2.0 * cos2mulphi1add2multheta;
        float tmp7 = cos2mulphi2add10multheta * sin2mulphi2add10multheta;
        float tmp8 = 20.0 * tmp7;
        float tmp9 = cos2mulphi2add10multheta * cos2mulphi2add10multheta;
        float tmp10 = 10.0 * tmp9;
        float tmp11 = 10.0 * (sin2mulphi2add10multheta * sin2mulphi2add10multheta);
        float tmp12 = 10.0 * cos2mulphi2add10multheta;

        numenator_coffs(i, 0) = 0.0;
        numenator_coffs(i, 1) = -tmp1;
        numenator_coffs(i, 2) = tmp3 - tmp5;
        numenator_coffs(i, 3) = tmp6;
        numenator_coffs(i, 4) = -tmp8;
        numenator_coffs(i, 5) = -tmp1 * tmp9 - tmp2 * tmp8;
        numenator_coffs(i, 6) = -tmp0 * tmp8 + 2.0 * tmp2 * tmp9 - tmp5 * tmp9;
        numenator_coffs(i, 7) = -sin2mulphi1add2multheta * tmp8 + tmp6 * tmp9;
        numenator_coffs(i, 8) = tmp10 - tmp11;
        numenator_coffs(i, 9) = -tmp1 * tmp7 - tmp11 * tmp2 + 10.0 * tmp2 * tmp9;
        numenator_coffs(i, 10) = tmp0 * tmp10 - tmp0 * tmp11 + tmp3 * tmp7 - tmp5 * tmp7;
        numenator_coffs(i, 11) = sin2mulphi1add2multheta * tmp10 - sin2mulphi1add2multheta * tmp11 + tmp6 * tmp7;
        numenator_coffs(i, 12) = -tmp12;
        numenator_coffs(i, 13) = 4.0 * cos2mulphi1add2multheta * sin2mulphi1add2multheta * sin2mulphi2add10multheta - tmp12 * tmp2;
        numenator_coffs(i, 14) = -sin2mulphi2add10multheta * tmp3 + 2.0 * sin2mulphi2add10multheta * tmp4 - tmp0 * tmp12;
        numenator_coffs(i, 15) = -sin2mulphi1add2multheta * tmp12 - sin2mulphi2add10multheta * tmp6;

        denominator_coffs(i, 0) = 1;
        denominator_coffs(i, 1) = tmp2;
        denominator_coffs(i, 2) = tmp0;
        denominator_coffs(i, 3) = sin2mulphi1add2multheta;
        denominator_coffs(i, 4) = tmp9;
        denominator_coffs(i, 5) = tmp2 * tmp9;
        denominator_coffs(i, 6) = tmp0 * tmp9;
        denominator_coffs(i, 7) = sin2mulphi1add2multheta * tmp9;
        denominator_coffs(i, 8) = tmp7;
        denominator_coffs(i, 9) = tmp2 * tmp7;
        denominator_coffs(i, 10) = tmp0 * tmp7;
        denominator_coffs(i, 11) = sin2mulphi1add2multheta * tmp7;
        denominator_coffs(i, 12) = -sin2mulphi2add10multheta;
        denominator_coffs(i, 13) = -sin2mulphi2add10multheta * tmp2;
        denominator_coffs(i, 14) = -sin2mulphi2add10multheta * tmp0;
        denominator_coffs(i, 15) = -sin2mulphi1add2multheta * sin2mulphi2add10multheta;
    }

    return {numenator_coffs, denominator_coffs};
}
