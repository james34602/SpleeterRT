void im2col_dilated_cpu(float* data_im, int channels, int height, int width, int ksize, int stride, int pad, int dilate_rate, float* data_col, int offsetH, int offsetW);
void col2im_dilated_cpu(float* data_col, int channels, int height, int width, int ksize, int stride, int pad, float* data_im, int offset);
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);
int conv_out_dim(int dim, int kernelSize, int stride, int pad, int dilate_rate);
int transpconv_out_dim(int dim, int kernelSize, int stride, int pad);