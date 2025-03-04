#include <mex.h>
#include <stdint.h>

//
// [counts, mean] = lnt_mean_tau(pixel, channels, x, y, t)
//
static const char *mexMain(int nlhs, mxArray *plhs[], int nrhs,
                           const mxArray *prhs[]) {
  // check arguments
  if (nlhs != 2)
    return "exactly 2 output arguments are required";

  if (nrhs != 5)
    return "exactly 5 input arguments are required";

  size_t pixels = (size_t)mxGetScalar(prhs[0]);
  double *channels = (double *)mxGetPr(prhs[1]);
  uint16_t channel1 = (uint16_t)channels[0];
  uint16_t channel2 = (uint16_t)channels[1];

  uint16_t *x = (uint16_t *)mxGetPr(prhs[2]);
  uint16_t *y = (uint16_t *)mxGetPr(prhs[3]);
  uint16_t *t = (uint16_t *)mxGetPr(prhs[4]);
  size_t length = (size_t)(mxGetM(prhs[2]) * mxGetN(prhs[2]));

  plhs[0] = mxCreateNumericMatrix(pixels, pixels, mxDOUBLE_CLASS, mxREAL);
  plhs[1] = mxCreateNumericMatrix(pixels, pixels, mxDOUBLE_CLASS, mxREAL);

  double *counts = (double *)mxGetPr(plhs[0]);
  double *mean = (double *)mxGetPr(plhs[1]);

  for (size_t i = 0; i < length; i++) {
    size_t xi = (size_t)x[i];
    size_t yi = (size_t)y[i];
    size_t ti = (size_t)t[i];

    xi--;
    yi--;
    if (xi < pixels && yi < pixels && channel1 <= ti && ti <= channel2) {
      size_t j = yi * pixels + xi;
      counts[j] += 1.0;
      mean[j] += ti;
    }
  }

  return 0;
}

extern "C" void
#if (!defined WIN32)
    __attribute__((visibility("default")))
#endif
    mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  const char *error;

  mexLock();
  error = mexMain(nlhs, plhs, nrhs, prhs);
  mexUnlock();

  if (error) {
    mexErrMsgTxt(error);
  }
}
