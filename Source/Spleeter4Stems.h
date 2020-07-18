#define FFTSIZE 4096
#define ANALYSIS_OVERLAP 4
#define OVPSIZE (FFTSIZE / ANALYSIS_OVERLAP)
#define OUTPUTSEG ((OVPSIZE >> 1) << 1)
#define SAMPLESHIFT (FFTSIZE - (OVPSIZE << 1))
#define MINUSFFTSIZE (FFTSIZE - 1)
#define HALFWNDLEN ((FFTSIZE >> 1) + 1)
#define MAX_OUTPUT_BUFFERS 2
#define LATENCY ((OVPSIZE << 1) - OUTPUTSEG)
#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
#define COMPONENTS 8
#include "cpthread.h"
enum pt_state
{
	SETUP,
	IDLE,
	WORKING,
	GET_OFF_FROM_WORK
};
#define TASK_NB 5
#include "spleeter.h"
typedef struct
{
	enum pt_state state;
	pthread_cond_t work_cond;
	pthread_mutex_t work_mtx;
	pthread_cond_t boss_cond;
	pthread_mutex_t boss_mtx;
	void *parentStruct;
	int pos;
} pt_infoSpleeter4s;
typedef struct
{
	// Constant
	unsigned int mBitRev[FFTSIZE];
	float analysisWnd[FFTSIZE];
	float synthesisWnd[FFTSIZE];
	float 	mSineTab[FFTSIZE];
	// Shared variable between all FFT length and modes and channel config
	int  mOutputReadSampleOffset;
	int  mOutputBufferCount; // How many buffers are actually in use
	float 	*mOutputBuffer[MAX_OUTPUT_BUFFERS];
	float buffer[MAX_OUTPUT_BUFFERS][OVPSIZE * COMPONENTS];
	unsigned int mInputSamplesNeeded;
	unsigned int mInputPos;
	float 	mInput[2][FFTSIZE];
	float 	mOverlapStage2dash[COMPONENTS][OUTPUTSEG];
	float 	mTempLBuffer[FFTSIZE];
	float 	mTempRBuffer[FFTSIZE];
	float timeDomainOut[COMPONENTS][FFTSIZE];
	float *signalIOPointer[COMPONENTS];
	// Neural network
	spleeter nn[4];
	float *maskPtr[2][4];
	float *tmpSpectro, *magnitudeSpectrogram, *complexSpectrogram[2][4];
	int analyseBinLimit;
	int timeStep;
	int nnMaskCursor;
	int outputFramePtr;
	// IFFT Threads
	pthread_t threads[TASK_NB];
	pt_infoSpleeter4s shared_info[TASK_NB];
	int needWait;
} Spleeter4Stems;
void Spleeter4StemsInit(Spleeter4Stems *msr, int initSpectralBinLimit, int initTimeStep, void *coeffProvider[4]);
void Spleeter4StemsFree(Spleeter4Stems *msr);
void Spleeter4StemsProcessSamples(Spleeter4Stems *msr, const float *inLeft, const float *inRight, int inSampleCount, float **components);