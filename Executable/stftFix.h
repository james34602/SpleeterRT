#ifndef _STFT_H_
#define _STFT_H_
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
enum pt_state
{
	SETUP,
	IDLE,
	WORKING,
	GET_OFF_FROM_WORK
};
#define FFTSIZE 4096
#define LAP 4
#define HOPSIZE (FFTSIZE / LAP)
#define HALFWNDLEN ((FFTSIZE >> 1) + 1)
#define NPTDIV2 (FFTSIZE >> 2)
typedef struct
{
	unsigned int mBitRev[FFTSIZE];
	float 	mPreWindow[FFTSIZE];
	float 	mPostWindow[FFTSIZE];
	float 	mSineTab[FFTSIZE];
	void *threads;
	void *stftThreadData;
	void *istftThreadData;
	size_t targetCore;
	float **_data[2];
	void *shared_info;
} OfflineSTFT;
void InitSTFT(OfflineSTFT *st, size_t targetCore);
void FreeSTFT(OfflineSTFT *st);
size_t stft(OfflineSTFT *st, const float *dataL, const float *dataR, size_t data_size, float **resultLRe, float **resultLIm, float **resultRRe, float **resultRIm);
size_t istft(OfflineSTFT *st, float *dataLRe, float *dataLIm, float *dataRRe, float *dataRIm, size_t data_size, float **resultL, float **resultR);
#ifdef __cplusplus
}
#endif
#endif