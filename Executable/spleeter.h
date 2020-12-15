#include <stddef.h>
#include <stdint.h>
#define TBL_SIZE (1025)
#define TBL_SIZE_MINUS1 (TBL_SIZE - 1)
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
	uint16_t down1_convWeight[5 * 5 * 2 * 16]; uint16_t down1_convBias[16];
	uint16_t down1_batchNorm[16 * 2];
	uint16_t down2_convWeight[5 * 5 * 16 * 32]; uint16_t down2_convBias[32];
	uint16_t down2_batchNorm[32 * 2];
	uint16_t down3_convWeight[5 * 5 * 32 * 64]; uint16_t down3_convBias[64];
	uint16_t down3_batchNorm[64 * 2];
	uint16_t down4_convWeight[5 * 5 * 64 * 128]; uint16_t down4_convBias[128];
	uint16_t down4_batchNorm[128 * 2];
	uint16_t down5_convWeight[5 * 5 * 128 * 256]; uint16_t down5_convBias[256];
	uint16_t down5_batchNorm[256 * 2];
	uint16_t down6_convWeight[5 * 5 * 256 * 512]; uint16_t down6_convBias[512];
	uint16_t up1_transp_convWeight[5 * 5 * 256 * 512]; uint16_t up1_transp_convBias[256];
	uint16_t up1_batchNorm[256 * 2];
	uint16_t up2_transp_convWeight[5 * 5 * 128 * 512]; uint16_t up2_transp_convBias[128];
	uint16_t up2_batchNorm[128 * 2];
	uint16_t up3_transp_convWeight[5 * 5 * 64 * 256]; uint16_t up3_transp_convBias[64];
	uint16_t up3_batchNorm[64 * 2];
	uint16_t up4_transp_convWeight[5 * 5 * 32 * 128]; uint16_t up4_transp_convBias[32];
	uint16_t up4_batchNorm[32 * 2];
	uint16_t up5_transp_convWeight[5 * 5 * 16 * 64]; uint16_t up5_transp_convBias[16];
	uint16_t up5_batchNorm[16 * 2];
	uint16_t up6_transp_convWeight[5 * 5 * 1 * 32]; uint16_t up6_transp_convBias[1];
	uint16_t up6_batchNorm[1 * 2];
	uint16_t up7_convWeight[4 * 4 * 1 * 2]; uint16_t up7_convBias[2];
} spleeterQuantizedSubNet;
typedef struct
{
	spleeterQuantizedSubNet net[2];
} spleeterQuantized;
typedef struct _spleeter *spleeter;
size_t getCoeffSize();
void* allocateSpleeterStr();
void initSpleeter(struct _spleeter *nn, size_t width, size_t height, int stemMode, void *coeff);
void getMaskPtr(struct _spleeter *nn, float **mask);
void freeSpleeter(struct _spleeter *nn);
void processSpleeter(struct _spleeter *nn, float *x, float *y);
