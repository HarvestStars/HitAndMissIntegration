#ifndef ORTHO_SAMPLING_H
#define ORTHO_SAMPLING_H

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

EXPORT void ortho_sampling_generate(int major, int runs, double *points_real, double *points_imag);

#endif // ORTHO_SAMPLING_H