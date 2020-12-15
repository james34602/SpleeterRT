void im2col_dilated_cpu(float* data_im, int channels, int height, int width, int ksize, int stride, int pad, int dilate_rate, float* data_col, int offsetH, int offsetW);
void col2im_dilated_cpu(float* data_col, int channels, int height, int width, int ksize, int stride, int pad, float* data_im, int offsetH, int offsetW);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);
void gemv(const int M, const int N, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
int conv_out_dim(int dim, int kernelSize, int stride, int pad, int dilate_rate);
int transpconv_out_dim(int dim, int kernelSize, int stride, int pad, int outputPadding);