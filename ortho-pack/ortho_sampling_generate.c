#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif
#include "mt19937.h"
#include "rand_support.h"
#include "ortho_sampling_generate.h"

#define SAMPLES(major) ((major) * (major))

void ortho_sampling_generate(int major, int runs, double min_bound_real, double max_bound_real, double min_bound_imag, double max_bound_imag, double *points_real, double *points_imag)
{
    int i, j, k, m;
    long double range_real = max_bound_real - min_bound_real;
    long double range_imag = max_bound_imag - min_bound_imag;
    long double scale_real = range_real / ((long double) SAMPLES(major));
    long double scale_imag = range_imag / ((long double) SAMPLES(major));
    double x, y;

    long **xlist = malloc(major * sizeof(long *));
    long **ylist = malloc(major * sizeof(long *));
    for (i = 0; i < major; i++) {
        xlist[i] = malloc(major * sizeof(long));
        ylist[i] = malloc(major * sizeof(long));
    }

    init_genrand(3737);
    m = 0;

    // init xlist and ylist
    for (i = 0; i < major; i++) {
        for (j = 0; j < major; j++) {
            xlist[i][j] = ylist[i][j] = m++;
        }
    }

    int point_index = 0;

    // repeat RUNS times
    for (k = 0; k < runs; k++) {
        for (i = 0; i < major; i++) {
            // permute xlist[i] and ylist[i]
            permute(xlist[i], major);
            permute(ylist[i], major);
        }
        for (i = 0; i < major; i++) {  // sub-square column
            for (j = 0; j < major; j++) {  // sub-square row
                // generate x coordinate
                x = min_bound_real + scale_real * (xlist[i][j] + (long double) genrand_real2());
                // generate y coordinate
                y = min_bound_imag + scale_imag * (ylist[j][i] + (long double) genrand_real2());

                // store the real and imaginary parts of the point
                points_real[point_index] = x;
                points_imag[point_index] = y;
                point_index++;
            }
        }
    }

    // clean up
    for (i = 0; i < major; i++) {
        free(xlist[i]);
        free(ylist[i]);
    }
    free(xlist);
    free(ylist);
}
