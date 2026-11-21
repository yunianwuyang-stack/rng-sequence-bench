#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// x is participant by trial by alternative by coefficient, in row-major order.
extern "C" void subject_log_likelihood(const double* x, const int* y,
                                      const double* beta, double* output,
                                      int participants, int trials, int options, int dimension) {
    std::vector<double> logits(options);
    for (int i = 0; i < participants; ++i) {
        const auto* coefficient = beta + static_cast<std::size_t>(i) * dimension;
        double result = 0.;
        for (int t = 0; t < trials; ++t) {
            for (int j = 0; j < options; ++j) {
                double logit = 0.;
                for (int d = 0; d < dimension; ++d)
                    logit += *x++ * coefficient[d];
                logits[j] = logit;
            }
            const double highest = *std::max_element(logits.begin(), logits.end());
            double denominator = 0.;
            for (const auto value : logits)
                denominator += std::exp(value - highest);
            result += logits[*y++] - highest - std::log(denominator);
        }
        output[i] = result;
    }
}
