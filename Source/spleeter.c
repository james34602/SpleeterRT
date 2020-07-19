#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "spleeter.h"
typedef struct
{
	float down1_convWeight[5 * 5 * 2 * 16]; float down1_convBias[16];
	float down1_batchNorm[16 * 2];
	float down2_convWeight[5 * 5 * 16 * 32]; float down2_convBias[32];
	float down2_batchNorm[32 * 2];
	float down3_convWeight[5 * 5 * 32 * 64]; float down3_convBias[64];
	float down3_batchNorm[64 * 2];
	float down4_convWeight[5 * 5 * 64 * 128]; float down4_convBias[128];
	float down4_batchNorm[128 * 2];
	float down5_convWeight[5 * 5 * 128 * 256]; float down5_convBias[256];
	float down5_batchNorm[256 * 2];
	float down6_convWeight[5 * 5 * 256 * 512]; float down6_convBias[512];
	float up1_transp_convWeight[5 * 5 * 256 * 512]; float up1_transp_convBias[256];
	float up1_batchNorm[256 * 2];
	float up2_transp_convWeight[5 * 5 * 128 * 512]; float up2_transp_convBias[128];
	float up2_batchNorm[128 * 2];
	float up3_transp_convWeight[5 * 5 * 64 * 256]; float up3_transp_convBias[64];
	float up3_batchNorm[64 * 2];
	float up4_transp_convWeight[5 * 5 * 32 * 128]; float up4_transp_convBias[32];
	float up4_batchNorm[32 * 2];
	float up5_transp_convWeight[5 * 5 * 16 * 64]; float up5_transp_convBias[16];
	float up5_batchNorm[16 * 2];
	float up6_transp_convWeight[5 * 5 * 1 * 32]; float up6_transp_convBias[1];
	float up6_batchNorm[1 * 2];
	float up7_convWeight[4 * 4 * 1 * 2]; float up7_convBias[2];
} spleeterCoeff;
typedef struct
{
	int inputCh, outputCh, height, width, kernelSize, stride, pad, dilation, out_h, out_w, hoffset, woffset;
	float *kernel;
} DLConv;
typedef struct
{
	int inputCh, outputCh, height, width, stride, out_h, out_w, kernelSize;
	float *kernel;
} DLtransposedConv;
struct _spleeter
{
	spleeterCoeff *coeffProvider;
	DLConv convLy[7];
	DLtransposedConv tpconv[6];
	int whProd, whProd2, whProd3;
	int whProd4, whProd5, whProd6, whProd7;
	float *conv1, *conv2, *conv3, *conv4, *conv5, *conv6, *catLayer, *workspace;
	float(*activationEncoder)(float);
	float(*activationDecoder)(float);
};
float sigmoid(float x)
{
    if (x >= 0.0f)
        return 1.0f / (1.0f + expf(-x));
    else
    {
        float tmp = expf(x);
        return tmp / (1.0f + tmp);
    }
}
float leakyReLU(float x)
{
    return x >= 0.0f ? x : 0.2f * x;
}
float ReLU(float x)
{
    return x >= 0.0f ? x : 0.0f;
}
float ELU(float x)
{
	return x >= 0.0f ? x : expf(x) - 1.0f;
}
#include "conv_util.h"
void initTransposeConv2dLayer(DLtransposedConv *ly, int inputCh, int outputCh, int height, int width, int stride, float *kernel, int kernelSize)
{
	ly->inputCh = inputCh;
	ly->outputCh = outputCh;
	ly->height = height;
	ly->width = width;
	ly->stride = stride;
	ly->kernel = kernel;
	ly->kernelSize = kernelSize;
	ly->out_h = transpconv_out_dim(height, kernelSize, stride, stride);
	ly->out_w = transpconv_out_dim(width, kernelSize, stride, stride);
}
void processTransposeConv2dLayer(DLtransposedConv *ly, float *x, float *y, float *workspace)
{
	gemm(1, 0, ly->kernelSize * ly->kernelSize * ly->outputCh, ly->height * ly->width, ly->inputCh, 1.0f, ly->kernel, ly->kernelSize * ly->kernelSize * ly->outputCh, x, ly->height * ly->width, 0.0f, workspace, ly->height * ly->width);
	memset(y, 0, ly->out_w * ly->out_h * ly->outputCh * sizeof(float));
	col2im_dilated_cpu(workspace, ly->outputCh, ly->out_h, ly->out_w, ly->kernelSize, ly->stride, ly->stride, y, 1);
}
void initConv2dLayer(DLConv *ly, int inputCh, int outputCh, int height, int width, int stride, int padding, int dilation, float *kernel, int kernelSize, int hoffset, int woffset)
{
	ly->inputCh = inputCh;
	ly->outputCh = outputCh;
	ly->height = height;
	ly->width = width;
	ly->stride = stride;
	ly->kernel = kernel;
	ly->kernelSize = kernelSize;
	ly->dilation = dilation;
	ly->hoffset = hoffset;
	ly->woffset = woffset;
	ly->pad = padding + dilation - 1;
	ly->out_h = conv_out_dim(height, kernelSize, stride, ly->pad, dilation);
	ly->out_w = conv_out_dim(width, kernelSize, stride, ly->pad, dilation);
	//int samePadding = (int)(((float)stride * (((float)height / (float)stride) + ((float)kernelSize - (float)height + ((float)dilation - 1.0f) * ((float)kernelSize + 1.0f)) / (float)stride - 1.0f)) / 2.0f);
}
void processConv2dLayer(DLConv *ly, float *x, float *y, float *workspace)
{
	im2col_dilated_cpu(x, ly->inputCh, ly->height, ly->width, ly->kernelSize, ly->stride, ly->pad, ly->dilation, workspace, ly->hoffset, ly->woffset);
	gemm(0, 0, ly->outputCh, ly->out_h * ly->out_w, ly->kernelSize * ly->kernelSize * ly->inputCh, 1.0f, ly->kernel, ly->kernelSize * ly->kernelSize * ly->inputCh, workspace, ly->out_h * ly->out_w, 0.0f, y, ly->out_h * ly->out_w);
}
int maxArray(int *x, int n)
{
	int maxVal = x[0];
	for (int i = 1; i < n; i++)
	{
		if (x[i] > maxVal)
			maxVal = x[i];
	}
	return maxVal;
}
void initSpleeter(struct _spleeter *nn, int width, int height, int stemMode, void *coeff)
{
	nn->whProd = width * height;
	nn->whProd2 = width / 2 * height / 2;
	nn->whProd3 = width / 4 * height / 4;
	nn->whProd4 = width / 8 * height / 8;
	nn->whProd5 = width / 16 * height / 16;
	nn->whProd6 = width / 32 * height / 32;
	nn->whProd7 = width / 64 * height / 64;
	size_t largestSize = width * height * 32;
	const int kernelSize = 5;
	nn->conv1 = (float*)malloc(nn->whProd2 * 16 * sizeof(float));
	nn->conv2 = (float*)malloc(nn->whProd3 * 32 * sizeof(float));
	nn->conv3 = (float*)malloc(nn->whProd4 * 64 * sizeof(float));
	nn->conv4 = (float*)malloc(nn->whProd5 * 128 * sizeof(float));
	nn->conv5 = (float*)malloc(nn->whProd6 * 256 * sizeof(float));
	nn->conv6 = (float*)malloc(nn->whProd7 * 512 * sizeof(float));
	nn->catLayer = (float*)malloc(largestSize * sizeof(float));
	nn->coeffProvider = (spleeterCoeff*)coeff;
	if (!stemMode)
	{
		nn->activationEncoder = leakyReLU;
		nn->activationDecoder = ReLU;
	}
	else
	{
		nn->activationEncoder = ELU;
		nn->activationDecoder = ELU;
	}
	int stride = 2;
	int dilation = 1;
	int padding = 2;
	int workspaceMemReq[13];
	initConv2dLayer(&nn->convLy[0], 2, 16, height, width, stride, padding, dilation, nn->coeffProvider->down1_convWeight, 5, 2, 2);
	initConv2dLayer(&nn->convLy[1], 16, 32, nn->convLy[0].out_h, nn->convLy[0].out_w, stride, padding, dilation, nn->coeffProvider->down2_convWeight, 5, 2, 2);
	initConv2dLayer(&nn->convLy[2], 32, 64, nn->convLy[1].out_h, nn->convLy[1].out_w, stride, padding, dilation, nn->coeffProvider->down3_convWeight, 5, 2, 2);
	initConv2dLayer(&nn->convLy[3], 64, 128, nn->convLy[2].out_h, nn->convLy[2].out_w, stride, padding, dilation, nn->coeffProvider->down4_convWeight, 5, 2, 2);
	initConv2dLayer(&nn->convLy[4], 128, 256, nn->convLy[3].out_h, nn->convLy[3].out_w, stride, padding, dilation, nn->coeffProvider->down5_convWeight, 5, 2, 2);
	initConv2dLayer(&nn->convLy[5], 256, 512, nn->convLy[4].out_h, nn->convLy[4].out_w, stride, padding, dilation, nn->coeffProvider->down6_convWeight, 5, 2, 2);
	initTransposeConv2dLayer(&nn->tpconv[0], 512, 256, nn->convLy[5].out_h, nn->convLy[5].out_w, stride, nn->coeffProvider->up1_transp_convWeight, kernelSize);
	initTransposeConv2dLayer(&nn->tpconv[1], 512, 128, nn->tpconv[0].out_h, nn->tpconv[0].out_w, stride, nn->coeffProvider->up2_transp_convWeight, kernelSize);
	initTransposeConv2dLayer(&nn->tpconv[2], 256, 64, nn->tpconv[1].out_h, nn->tpconv[1].out_w, stride, nn->coeffProvider->up3_transp_convWeight, kernelSize);
	initTransposeConv2dLayer(&nn->tpconv[3], 128, 32, nn->tpconv[2].out_h, nn->tpconv[2].out_w, stride, nn->coeffProvider->up4_transp_convWeight, kernelSize);
	initTransposeConv2dLayer(&nn->tpconv[4], 64, 16, nn->tpconv[3].out_h, nn->tpconv[3].out_w, stride, nn->coeffProvider->up5_transp_convWeight, kernelSize);
	initTransposeConv2dLayer(&nn->tpconv[5], 32, 1, nn->tpconv[4].out_h, nn->tpconv[4].out_w, stride, nn->coeffProvider->up6_transp_convWeight, kernelSize);
	initConv2dLayer(&nn->convLy[6], 1, 2, nn->tpconv[5].out_h, nn->tpconv[5].out_w, 1, 3, 2, nn->coeffProvider->up7_convWeight, 4, 1, 1);
	workspaceMemReq[0] = nn->convLy[0].width * nn->convLy[0].height * nn->convLy[0].kernelSize * nn->convLy[0].kernelSize * nn->convLy[0].inputCh;
	workspaceMemReq[1] = nn->convLy[1].width * nn->convLy[1].height * nn->convLy[1].kernelSize * nn->convLy[1].kernelSize * nn->convLy[1].inputCh;
	workspaceMemReq[2] = nn->convLy[2].width * nn->convLy[2].height * nn->convLy[2].kernelSize * nn->convLy[2].kernelSize * nn->convLy[2].inputCh;
	workspaceMemReq[3] = nn->convLy[3].width * nn->convLy[3].height * nn->convLy[3].kernelSize * nn->convLy[3].kernelSize * nn->convLy[3].inputCh;
	workspaceMemReq[4] = nn->convLy[4].width * nn->convLy[4].height * nn->convLy[4].kernelSize * nn->convLy[4].kernelSize * nn->convLy[4].inputCh;
	workspaceMemReq[5] = nn->convLy[5].width * nn->convLy[5].height * nn->convLy[5].kernelSize * nn->convLy[5].kernelSize * nn->convLy[5].inputCh;
	workspaceMemReq[6] = nn->tpconv[0].out_h * nn->tpconv[0].out_w * kernelSize * kernelSize * nn->tpconv[0].outputCh;
	workspaceMemReq[7] = nn->tpconv[1].out_h * nn->tpconv[1].out_w * kernelSize * kernelSize * nn->tpconv[1].outputCh;
	workspaceMemReq[8] = nn->tpconv[2].out_h * nn->tpconv[2].out_w * kernelSize * kernelSize * nn->tpconv[2].outputCh;
	workspaceMemReq[9] = nn->tpconv[3].out_h * nn->tpconv[3].out_w * kernelSize * kernelSize * nn->tpconv[3].outputCh;
	workspaceMemReq[10] = nn->tpconv[4].out_h * nn->tpconv[4].out_w * kernelSize * kernelSize * nn->tpconv[4].outputCh;
	workspaceMemReq[11] = nn->tpconv[5].out_h * nn->tpconv[5].out_w * kernelSize * kernelSize * nn->tpconv[5].outputCh;
	workspaceMemReq[12] = nn->convLy[6].width * nn->convLy[6].height * nn->convLy[6].kernelSize * nn->convLy[6].kernelSize * nn->convLy[6].inputCh;
	int maxWorkspaceSize = maxArray(workspaceMemReq, 13);
	nn->workspace = (float*)malloc(maxWorkspaceSize * sizeof(float));
}
void* allocateSpleeterStr()
{
	return (void*)malloc(sizeof(struct _spleeter));
}
void processSpleeter(struct _spleeter *nn, float *x, float *y)
{
	int s, i;
	float val;
	processConv2dLayer(&nn->convLy[0], x, nn->conv1, nn->workspace);
	for (s = 0; s < 16; s++)
	{
		for (i = 0; i < nn->whProd2; i++)
		{
			val = nn->conv1[s * nn->whProd2 + i] + nn->coeffProvider->down1_convBias[s];
			nn->conv1[s * nn->whProd2 + i] = val;
			nn->catLayer[s * nn->whProd2 + i] = nn->activationEncoder(nn->coeffProvider->down1_batchNorm[16 * 1 + s] * val + nn->coeffProvider->down1_batchNorm[s]);
		}
	}
	processConv2dLayer(&nn->convLy[1], nn->catLayer, nn->conv2, nn->workspace);
	for (s = 0; s < 32; s++)
	{
		for (i = 0; i < nn->whProd3; i++)
		{
			val = nn->conv2[s * nn->whProd3 + i] + nn->coeffProvider->down2_convBias[s];
			nn->conv2[s * nn->whProd3 + i] = val;
			nn->catLayer[s * nn->whProd3 + i] = nn->activationEncoder(nn->coeffProvider->down2_batchNorm[32 + s] * val + nn->coeffProvider->down2_batchNorm[s]);
		}
	}
	processConv2dLayer(&nn->convLy[2], nn->catLayer, nn->conv3, nn->workspace);
	for (s = 0; s < 64; s++)
	{
		for (i = 0; i < nn->whProd4; i++)
		{
			val = nn->conv3[s * nn->whProd4 + i] + nn->coeffProvider->down3_convBias[s];
			nn->conv3[s * nn->whProd4 + i] = val;
			nn->catLayer[s * nn->whProd4 + i] = nn->activationEncoder(nn->coeffProvider->down3_batchNorm[64 + s] * val + nn->coeffProvider->down3_batchNorm[s]);
		}
	}
	processConv2dLayer(&nn->convLy[3], nn->catLayer, nn->conv4, nn->workspace);
	for (s = 0; s < 128; s++)
	{
		for (i = 0; i < nn->whProd5; i++)
		{
			val = nn->conv4[s * nn->whProd5 + i] + nn->coeffProvider->down4_convBias[s];
			nn->conv4[s * nn->whProd5 + i] = val;
			nn->catLayer[s * nn->whProd5 + i] = nn->activationEncoder(nn->coeffProvider->down4_batchNorm[128 + s] * val + nn->coeffProvider->down4_batchNorm[s]);
		}
	}
	processConv2dLayer(&nn->convLy[4], nn->catLayer, nn->conv5, nn->workspace);
	for (s = 0; s < 256; s++)
	{
		for (i = 0; i < nn->whProd6; i++)
		{
			val = nn->conv5[s * nn->whProd6 + i] + nn->coeffProvider->down5_convBias[s];
			nn->conv5[s * nn->whProd6 + i] = val;
			nn->catLayer[s * nn->whProd6 + i] = nn->activationEncoder(nn->coeffProvider->down5_batchNorm[256 + s] * val + nn->coeffProvider->down5_batchNorm[s]);
		}
	}
	processConv2dLayer(&nn->convLy[5], nn->catLayer, nn->conv6, nn->workspace);
	for (s = 0; s < 512; s++)
	{
		for (i = 0; i < nn->whProd7; i++)
		{
			nn->conv6[s * nn->whProd7 + i] += nn->coeffProvider->down6_convBias[s];
		}
	}
	processTransposeConv2dLayer(&nn->tpconv[0], nn->conv6, nn->catLayer + 256 * nn->whProd6, nn->workspace);
	for (s = 0; s < 256; s++)
	{
		for (i = 0; i < nn->whProd6; i++)
		{
			val = nn->activationDecoder(nn->catLayer[(s + 256) * nn->whProd6 + i] + nn->coeffProvider->up1_transp_convBias[s]);
			nn->catLayer[(s + 256) * nn->whProd6 + i] = nn->coeffProvider->up1_batchNorm[256 + s] * val + nn->coeffProvider->up1_batchNorm[s];
		}
	}
	memcpy(nn->catLayer, nn->conv5, 256 * nn->whProd6 * sizeof(float));
	processTransposeConv2dLayer(&nn->tpconv[1], nn->catLayer, nn->catLayer + 128 * nn->whProd5, nn->workspace);
	for (s = 0; s < 128; s++)
	{
		for (i = 0; i < nn->whProd5; i++)
		{
			val = nn->activationDecoder(nn->catLayer[(s + 128) * nn->whProd5 + i] + nn->coeffProvider->up2_transp_convBias[s]);
			nn->catLayer[(s + 128) * nn->whProd5 + i] = nn->coeffProvider->up2_batchNorm[128 + s] * val + nn->coeffProvider->up2_batchNorm[s];
		}
	}
	memcpy(nn->catLayer, nn->conv4, 128 * nn->whProd5 * sizeof(float));
	processTransposeConv2dLayer(&nn->tpconv[2], nn->catLayer, nn->catLayer + 64 * nn->whProd4, nn->workspace);
	for (s = 0; s < 64; s++)
	{
		for (i = 0; i < nn->whProd4; i++)
		{
			val = nn->activationDecoder(nn->catLayer[(s + 64) * nn->whProd4 + i] + nn->coeffProvider->up3_transp_convBias[s]);
			nn->catLayer[(s + 64) * nn->whProd4 + i] = nn->coeffProvider->up3_batchNorm[64 + s] * val + nn->coeffProvider->up3_batchNorm[s];
		}
	}
	memcpy(nn->catLayer, nn->conv3, 64 * nn->whProd4 * sizeof(float));
	processTransposeConv2dLayer(&nn->tpconv[3], nn->catLayer, nn->catLayer + 32 * nn->whProd3, nn->workspace);
	for (s = 0; s < 32; s++)
	{
		for (i = 0; i < nn->whProd3; i++)
		{
			val = nn->activationDecoder(nn->catLayer[(s + 32) * nn->whProd3 + i] + nn->coeffProvider->up4_transp_convBias[s]);
			nn->catLayer[(s + 32) * nn->whProd3 + i] = nn->coeffProvider->up4_batchNorm[32 + s] * val + nn->coeffProvider->up4_batchNorm[s];
		}
	}
	memcpy(nn->catLayer, nn->conv2, 32 * nn->whProd3 * sizeof(float));
	processTransposeConv2dLayer(&nn->tpconv[4], nn->catLayer, nn->catLayer + 16 * nn->whProd2, nn->workspace);
	for (s = 0; s < 16; s++)
	{
		for (i = 0; i < nn->whProd2; i++)
		{
			val = nn->activationDecoder(nn->catLayer[(s + 16) * nn->whProd2 + i] + nn->coeffProvider->up5_transp_convBias[s]);
			nn->catLayer[(s + 16) * nn->whProd2 + i] = nn->coeffProvider->up5_batchNorm[16 + s] * val + nn->coeffProvider->up5_batchNorm[s];
		}
	}
	memcpy(nn->catLayer, nn->conv1, 16 * nn->whProd2 * sizeof(float));
	processTransposeConv2dLayer(&nn->tpconv[5], nn->catLayer, nn->catLayer, nn->workspace);
	for (i = 0; i < nn->whProd; i++)
	{
		val = nn->activationDecoder(nn->catLayer[i] + nn->coeffProvider->up6_transp_convBias[0]);
		nn->catLayer[i] = nn->coeffProvider->up6_batchNorm[1] * val + nn->coeffProvider->up6_batchNorm[0];
	}
	processConv2dLayer(&nn->convLy[6], nn->catLayer, nn->catLayer, nn->workspace);
	for (s = 0; s < 2; s++)
	{
		for (i = 0; i < nn->whProd; i++)
			y[s * nn->whProd + i] = sigmoid(nn->catLayer[s * nn->whProd + i] + nn->coeffProvider->up7_convBias[s]);
	}
}
size_t getCoeffSize()
{
	return sizeof(spleeterCoeff);
}
void getMaskPtr(struct _spleeter *nn, float **mask)
{
	*mask = (void*)nn->catLayer;
}
void freeSpleeter(struct _spleeter *nn)
{
	free(nn->conv1);
	free(nn->conv2);
	free(nn->conv3);
	free(nn->conv4);
	free(nn->conv5);
	free(nn->conv6);
	free(nn->catLayer);
	free(nn->workspace);
}
