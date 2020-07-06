#include "conv_util.h"
float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}
void im2col_dilated_cpu(float* data_im, int channels, int height, int width, int ksize, int stride, int pad, int dilate_rate, float* data_col, int offsetH, int offsetW)
{
    int c,h,w;
    int dilate_ksize = (dilate_rate - 1) * (ksize + 1) + ksize;
    int height_col = (height + 2*pad - dilate_ksize) / stride + 1;
    int width_col = (width + 2*pad - dilate_ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
	{
        int w_offset = c % ksize + 1 + offsetW;
        int h_offset = (c / ksize) % ksize + 1 + offsetH;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
		{
            for (w = 0; w < width_col; ++w)
			{
                int im_row = h_offset * dilate_rate + h * stride - dilate_rate - 1;
                int im_col = w_offset * dilate_rate + w * stride - dilate_rate - 1;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}
void col2im_add_pixel_dilated(float *im, int height, int width, int channels, int row, int col, int channel, int pad, float val)
{
	row -= pad;
	col -= pad;
	if ((row - 1) < 0 || (col - 1) < 0 || (row - 1) >= height || (col - 1) >= width)
		return;
	im[col - 1 + width * (row - 1 + height * channel)] += val;
}
void col2im_dilated_cpu(float* data_col, int channels, int height, int width, int ksize, int stride, int pad, float* data_im, int offset)
{
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c)
	{
		int w_offset = c % ksize + 1 + offset;
		int h_offset = (c / ksize) % ksize + 1 + offset;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h)
		{
			for (w = 0; w < width_col; ++w)
			{
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				float val = data_col[col_index];
				col2im_add_pixel_dilated(data_im, height, width, channels, im_row, im_col, c_im, pad, val);
			}
		}
	}
}
int conv_out_dim(int dim, int kernelSize, int stride, int pad, int dilate_rate)
{
	int dsize = (dilate_rate - 1) * (kernelSize + 1) + kernelSize;
	return (dim + 2 * pad - dsize) / stride + 1;
}
int transpconv_out_dim(int dim, int kernelSize, int stride, int pad)
{
	return (dim - 1) * stride - 2 * pad + kernelSize + 1;
}