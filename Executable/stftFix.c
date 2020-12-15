#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "codelet.h"
#include "stftFix.h"
size_t LLIntegerLog2(size_t v)
{
	size_t i = 0;
	while (v > 1)
	{
		++i;
		v >>= 1;
	}
	return i;
}
size_t LLRevBits(size_t x, size_t bits)
{
	size_t y = 0;
	while (bits--)
	{
		y = (y + y) + (x & 1);
		x >>= 1;
	}
	return y;
}
void LLraisedCosTbl(double *dst, int n, int windowSizePadded, int overlapCount)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	const double scalefac = 1.0 / windowSizePadded;
	double power = 1.0;
	if (overlapCount == 2)
		power = 0.5;
	for (int i = 0; i < n; ++i)
		dst[i] = scalefac * pow(0.5*(1.0 - cos(twopi_over_n * (i + 0.5))), power);
}
void LLsinHalfTbl(double *dst, int n)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	for (int i = 0; i < n; ++i)
		dst[i] = sin(twopi_over_n * i);
}
void LLbitReversalTbl(unsigned *dst, size_t n)
{
	size_t bits = LLIntegerLog2(n);
	for (size_t i = 0; i < n; ++i)
		dst[i] = (unsigned int)LLRevBits(i, bits);
}
void LLraisedCosTblFloat(float *dst, size_t n, size_t overlapCount)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	const double scalefac = 1.0 / n;
	float power = 1.0f;
	if (overlapCount == 2)
		power = 0.5f;
	for (size_t i = 0; i < n; ++i)
		dst[i] = (float)(scalefac * pow(0.5*(1.0 - cos(twopi_over_n * (i + 0.5))), power));
}
void LLsinHalfTblFloat(float *dst, size_t n)
{
	const double twopi_over_n = 6.283185307179586476925286766559 / n;
	for (int i = 0; i < n; ++i)
		dst[i] = (float)sin(twopi_over_n * i);
}
void LLCreatePostWindowFloat(float *dst, size_t windowSize, size_t overlapCount)
{
	const float powerIntegrals[8] = { 1.0f, 1.0f / 2.0f, 3.0f / 8.0f, 5.0f / 16.0f, 35.0f / 128.0f,
		63.0f / 256.0f, 231.0f / 1024.0f, 429.0f / 2048.0f };
	int power = 1;
	if (overlapCount == 2)
		power = 0;
	const float scalefac = (float)windowSize * (powerIntegrals[1] / powerIntegrals[power + 1]);
	LLraisedCosTblFloat(dst, windowSize, overlapCount);
	for (size_t i = 0; i < windowSize; ++i)
		dst[i] *= scalefac;
}
#include "cpthread.h"
typedef struct
{
	size_t rangeMin, rangeMax;
	float *_data[2];
	const float *dataL, *dataR;
	float *mPreWindow, *mSineTab;
	unsigned int *mBitRev;
	float *res[4];
	size_t startingPoint;
} STFTThread;
typedef struct
{
	size_t rangeMin, rangeMax;
	float *_data[2];
	float *dataLRe, *dataLIm, *dataRRe, *dataRIm;
	float *mPreWindow, *mSineTab;
	unsigned int *mBitRev;
} ISTFTThread;
typedef struct
{
	enum pt_state state;
	pthread_cond_t work_cond;
	pthread_mutex_t work_mtx;
	pthread_cond_t boss_cond;
	pthread_mutex_t boss_mtx;
	STFTThread *fwd;
	ISTFTThread *rev;
} STFTthreadpool;
void *task_STFT(void *arg)
{
	STFTthreadpool *info = (STFTthreadpool *)arg;
	STFTThread *rb = (STFTThread*)info->fwd;
	size_t pos, i, symIdx;
	float lR, lI, rR, rI;
	// cond_wait mutex must be locked before we can wait
	pthread_mutex_lock(&(info->work_mtx));
	//	printf("<worker %i> start\n", task);
		// ensure boss is waiting
	pthread_mutex_lock(&(info->boss_mtx));
	// signal to boss that setup is complete
	info->state = IDLE;
	// wake-up signal
	pthread_cond_signal(&(info->boss_cond));
	pthread_mutex_unlock(&(info->boss_mtx));
	while (1)
	{
		pthread_cond_wait(&(info->work_cond), &(info->work_mtx));
		if (GET_OFF_FROM_WORK == info->state)
			break; // kill thread
		if (IDLE == info->state)
			continue; // accidental wake-up
		// do blocking task
		size_t idx = rb->startingPoint;
		for (pos = rb->rangeMin; pos < rb->rangeMax; pos += HOPSIZE)
		{
			for (i = 0; i < FFTSIZE; ++i)
			{
				const float w = rb->mPreWindow[i];
				rb->_data[0][rb->mBitRev[i]] = rb->dataL[pos + i] * w;
				rb->_data[1][rb->mBitRev[i]] = rb->dataR[pos + i] * w;
			}
			DFT4096(rb->_data[0], rb->mSineTab);
			DFT4096(rb->_data[1], rb->mSineTab);
			rb->res[0][idx * FFTSIZE] = rb->_data[0][0] * 2.0f;
			rb->res[2][idx * FFTSIZE] = rb->_data[1][0] * 2.0f;
			rb->res[1][idx * FFTSIZE] = 0.0f;
			rb->res[3][idx * FFTSIZE] = 0.0f;
			for (i = 1; i < HALFWNDLEN; i++)
			{
				symIdx = FFTSIZE - i;
				lR = rb->_data[0][i] + rb->_data[0][symIdx];
				lI = rb->_data[0][i] - rb->_data[0][symIdx];
				rR = rb->_data[1][i] + rb->_data[1][symIdx];
				rI = rb->_data[1][i] - rb->_data[1][symIdx];
				rb->res[0][idx * FFTSIZE + i] = lR;
				rb->res[1][idx * FFTSIZE + i] = lI;
				rb->res[2][idx * FFTSIZE + i] = rR;
				rb->res[3][idx * FFTSIZE + i] = rI;
			}
			idx++;
		}
		// ensure boss is waiting
		pthread_mutex_lock(&(info->boss_mtx));
		// indicate that job is done
		info->state = IDLE;
		// wake-up signal
		pthread_cond_signal(&(info->boss_cond));
		pthread_mutex_unlock(&(info->boss_mtx));
	}
	pthread_mutex_unlock(&(info->work_mtx));
	pthread_exit(NULL);
	return 0;
}
void *task_ISTFT(void *arg)
{
	STFTthreadpool *info = (STFTthreadpool *)arg;
	ISTFTThread *rb = (ISTFTThread*)info->rev;
	size_t i, j, symIdx;
	unsigned int bitRevFwd, bitRevSym;
	// cond_wait mutex must be locked before we can wait
	pthread_mutex_lock(&(info->work_mtx));
	//	printf("<worker %i> start\n", task);
		// ensure boss is waiting
	pthread_mutex_lock(&(info->boss_mtx));
	// signal to boss that setup is complete
	info->state = IDLE;
	// wake-up signal
	pthread_cond_signal(&(info->boss_cond));
	pthread_mutex_unlock(&(info->boss_mtx));
	while (1)
	{
		pthread_cond_wait(&(info->work_cond), &(info->work_mtx));
		if (GET_OFF_FROM_WORK == info->state)
			break; // kill thread
		if (IDLE == info->state)
			continue; // accidental wake-up
		for (i = rb->rangeMin; i < rb->rangeMax; i++)
		{
			rb->_data[0][0] = rb->dataLRe[i * FFTSIZE];
			rb->_data[1][0] = rb->dataRRe[i * FFTSIZE];
			for (j = 1; j < HALFWNDLEN; j++)
			{
				symIdx = FFTSIZE - j;
				bitRevFwd = rb->mBitRev[j];
				bitRevSym = rb->mBitRev[symIdx];
				rb->_data[0][bitRevFwd] = rb->dataLRe[i * FFTSIZE + j] + rb->dataLIm[i * FFTSIZE + j];
				rb->_data[0][bitRevSym] = rb->dataLRe[i * FFTSIZE + j] - rb->dataLIm[i * FFTSIZE + j];
				rb->_data[1][bitRevFwd] = rb->dataRRe[i * FFTSIZE + j] + rb->dataRIm[i * FFTSIZE + j];
				rb->_data[1][bitRevSym] = rb->dataRRe[i * FFTSIZE + j] - rb->dataRIm[i * FFTSIZE + j];
			}
			DFT4096(rb->_data[0], rb->mSineTab);
			DFT4096(rb->_data[1], rb->mSineTab);
			memcpy(&rb->dataLRe[i * FFTSIZE], rb->_data[0], FFTSIZE * sizeof(float));
			memcpy(&rb->dataRRe[i * FFTSIZE], rb->_data[1], FFTSIZE * sizeof(float));
		}
		// ensure boss is waiting
		pthread_mutex_lock(&(info->boss_mtx));
		// indicate that job is done
		info->state = IDLE;
		// wake-up signal
		pthread_cond_signal(&(info->boss_cond));
		pthread_mutex_unlock(&(info->boss_mtx));
	}
	pthread_mutex_unlock(&(info->work_mtx));
	pthread_exit(NULL);
	return 0;
}
void task_start(OfflineSTFT *st, size_t task)
{
	STFTthreadpool *shared_info = st->shared_info;
	STFTthreadpool *info = &(shared_info[task]);
	// ensure worker is waiting
	pthread_mutex_lock(&(info->work_mtx));
	// set job information & state
	info->state = WORKING;
	// wake-up signal
	pthread_cond_signal(&(info->work_cond));
	pthread_mutex_unlock(&(info->work_mtx));
}
void task_wait(OfflineSTFT *st, size_t task)
{
	STFTthreadpool *shared_info = st->shared_info;
	STFTthreadpool *info = &(shared_info[task]);
	while (1)
	{
		pthread_cond_wait(&(info->boss_cond), &(info->boss_mtx));
		if (IDLE == info->state)
			break;
	}
}
void thread_initSpleeter4s(OfflineSTFT *st)
{
	int i;
	STFTthreadpool *shared_info = st->shared_info;
	pthread_t *threads = (pthread_t*)st->threads;
	for (i = 0; i < (st->targetCore - 1) << 1; i++)
	{
		//		GFT->sa_log_d(TAG, "Thread %d initialized", i + 1);
		STFTthreadpool *info = &(shared_info[i]);
		info->state = SETUP;
		pthread_cond_init(&(info->work_cond), NULL);
		pthread_mutex_init(&(info->work_mtx), NULL);
		pthread_cond_init(&(info->boss_cond), NULL);
		pthread_mutex_init(&(info->boss_mtx), NULL);
		pthread_mutex_lock(&(info->boss_mtx));
		if (i < (st->targetCore - 1))
		{
			STFTThread *ptrFwd = (STFTThread*)st->stftThreadData;
			info->fwd = &ptrFwd[i];
			info->rev = 0;
			pthread_create(&threads[i], NULL, task_STFT, (void *)info);
		}
		else
		{
			ISTFTThread *ptrRev = (ISTFTThread*)st->istftThreadData;
			info->fwd = 0;
			info->rev = &ptrRev[i - (st->targetCore - 1)];
			pthread_create(&threads[i], NULL, task_ISTFT, (void *)info);
		}
		task_wait(st, i);
	}
}
void thread_exitSpleeter4s(OfflineSTFT *st)
{
	STFTthreadpool *shared_info = st->shared_info;
	pthread_t *threads = (pthread_t*)st->threads;
	for (int i = 0; i < (st->targetCore - 1) << 1; i++)
	{
		//		GFT->sa_log_d(TAG, "Thread %d get off from work", i + 1);
		STFTthreadpool *info = &(shared_info[i]);
		// ensure the worker is waiting
		pthread_mutex_lock(&(info->work_mtx));
		info->state = GET_OFF_FROM_WORK;
		// wake-up signal
		pthread_cond_signal(&(info->work_cond));
		pthread_mutex_unlock(&(info->work_mtx));
		// wait for thread to exit
		pthread_join(threads[i], NULL);
		pthread_mutex_destroy(&(info->work_mtx));
		pthread_cond_destroy(&(info->work_cond));
		pthread_mutex_unlock(&(info->boss_mtx));
		pthread_mutex_destroy(&(info->boss_mtx));
		pthread_cond_destroy(&(info->boss_cond));
	}
}
void InitSTFT(OfflineSTFT *st, size_t targetCore)
{
	size_t i;
	LLbitReversalTbl(st->mBitRev, FFTSIZE);
	LLraisedCosTblFloat(st->mSineTab, FFTSIZE, LAP);
	for (i = 0; i < FFTSIZE; ++i)
		st->mPreWindow[i] = st->mSineTab[i] * (2.0f / (float)LAP);
	LLsinHalfTblFloat(st->mSineTab, FFTSIZE);
	LLCreatePostWindowFloat(st->mPostWindow, FFTSIZE, LAP);
	for (i = 0; i < FFTSIZE; ++i)
		st->mPostWindow[i] *= 0.5f;
	st->stftThreadData = malloc(targetCore * sizeof(STFTThread));
	st->istftThreadData = malloc(targetCore * sizeof(ISTFTThread));
	st->threads = malloc(((targetCore - 1) << 1) * sizeof(pthread_t));
	st->_data[0] = (float**)malloc(targetCore * sizeof(float*));
	st->_data[1] = (float**)malloc(targetCore * sizeof(float*));
	for (i = 0; i < targetCore; i++)
	{
		st->_data[0][i] = (float*)calloc(FFTSIZE, sizeof(float));
		st->_data[1][i] = (float*)calloc(FFTSIZE, sizeof(float));
	}
	STFTThread *stftThreadData = (STFTThread*)st->stftThreadData;
	ISTFTThread *istftThreadData = (ISTFTThread*)st->istftThreadData;
	st->targetCore = targetCore;
	for (i = 0; i < targetCore; i++)
	{
		stftThreadData[i]._data[0] = st->_data[0][i];
		stftThreadData[i]._data[1] = st->_data[1][i];
		stftThreadData[i].mPreWindow = st->mPreWindow;
		stftThreadData[i].mSineTab = st->mSineTab;
		stftThreadData[i].mBitRev = st->mBitRev;
		istftThreadData[i]._data[0] = st->_data[0][i];
		istftThreadData[i]._data[1] = st->_data[1][i];
		istftThreadData[i].mPreWindow = st->mPreWindow;
		istftThreadData[i].mSineTab = st->mSineTab;
		istftThreadData[i].mBitRev = st->mBitRev;
	}
	st->shared_info = malloc(((targetCore - 1) << 1) * sizeof(STFTthreadpool));
	thread_initSpleeter4s(st);
}
void FreeSTFT(OfflineSTFT *st)
{
	STFTthreadpool *shared_info = st->shared_info;
	for (int i = 0; i < (st->targetCore - 1) << 1; i++)
	{
		if (shared_info[i].state == WORKING)
			task_wait(st, i);
	}
	thread_exitSpleeter4s(st);
	free(st->shared_info);
	free(st->istftThreadData);
	free(st->stftThreadData);
	free(st->threads);
	for (size_t i = 0; i < st->targetCore; i++)
	{
		free(st->_data[0][i]);
		free(st->_data[1][i]);
	}
	free(st->_data[0]);
	free(st->_data[1]);
}
size_t stft(OfflineSTFT *st, const float *dataL, const float *dataR, size_t data_size, float **resultLRe, float **resultLIm, float **resultRRe, float **resultRIm)
{
	STFTThread *stftThreadData = (STFTThread*)st->stftThreadData;
	int i;
	size_t result_size = (size_t)ceil((double)data_size / (double)HOPSIZE);
    *resultLRe = (float *) calloc(result_size * FFTSIZE, sizeof(float));
	*resultLIm = (float *)calloc(result_size * FFTSIZE, sizeof(float));
	*resultRRe = (float *)calloc(result_size * FFTSIZE, sizeof(float));
	*resultRIm = (float *)calloc(result_size * FFTSIZE, sizeof(float));
	size_t symIdx;
	unsigned int bitRevFwd, bitRevSym;
	float lR, lI, rR, rI;
	size_t idx = 0;
	size_t pos = 0;
	size_t rangeM = ((data_size - FFTSIZE + HOPSIZE / LAP) / HOPSIZE) * HOPSIZE;
	size_t workloadMinor = (size_t)(ceilf(ceilf((float)rangeM / (float)st->targetCore) / (float)HOPSIZE) * HOPSIZE);
	if (st->targetCore > 1)
	{
		for (i = 0; i < st->targetCore; i++)
		{
			stftThreadData[i].rangeMin = i * workloadMinor;
			stftThreadData[i].rangeMax = i * workloadMinor + workloadMinor;
			stftThreadData[i].dataL = dataL;
			stftThreadData[i].dataR = dataR;
			stftThreadData[i].res[0] = *resultLRe;
			stftThreadData[i].res[1] = *resultLIm;
			stftThreadData[i].res[2] = *resultRRe;
			stftThreadData[i].res[3] = *resultRIm;
			stftThreadData[i].startingPoint = (workloadMinor / HOPSIZE) * i;
		}
		stftThreadData[st->targetCore - 1].rangeMin = stftThreadData[st->targetCore - 2].rangeMax;
		stftThreadData[st->targetCore - 1].rangeMax = rangeM;
		for (i = 0; i < st->targetCore - 1; i++)
			task_start(st, i);
		idx = stftThreadData[st->targetCore - 1].startingPoint;
		for (pos = stftThreadData[st->targetCore - 1].rangeMin; pos < stftThreadData[st->targetCore - 1].rangeMax; pos += HOPSIZE)
		{
			for (i = 0; i < FFTSIZE; ++i)
			{
				const float w = st->mPreWindow[i];
				st->_data[0][st->targetCore - 1][st->mBitRev[i]] = dataL[pos + i] * w;
				st->_data[1][st->targetCore - 1][st->mBitRev[i]] = dataR[pos + i] * w;
			}
			DFT4096(st->_data[0][st->targetCore - 1], st->mSineTab);
			DFT4096(st->_data[1][st->targetCore - 1], st->mSineTab);
			(*resultLRe)[idx * FFTSIZE] = st->_data[0][st->targetCore - 1][0] * 2.0f;
			(*resultRRe)[idx * FFTSIZE] = st->_data[1][st->targetCore - 1][0] * 2.0f;
			(*resultLIm)[idx * FFTSIZE] = 0.0f;
			(*resultRIm)[idx * FFTSIZE] = 0.0f;
			for (i = 1; i < HALFWNDLEN; i++)
			{
				symIdx = FFTSIZE - i;
				lR = st->_data[0][st->targetCore - 1][i] + st->_data[0][st->targetCore - 1][symIdx];
				lI = st->_data[0][st->targetCore - 1][i] - st->_data[0][st->targetCore - 1][symIdx];
				rR = st->_data[1][st->targetCore - 1][i] + st->_data[1][st->targetCore - 1][symIdx];
				rI = st->_data[1][st->targetCore - 1][i] - st->_data[1][st->targetCore - 1][symIdx];
				(*resultLRe)[idx * FFTSIZE + i] = lR;
				(*resultLIm)[idx * FFTSIZE + i] = lI;
				(*resultRRe)[idx * FFTSIZE + i] = rR;
				(*resultRIm)[idx * FFTSIZE + i] = rI;
			}
			idx++;
		}
		for (i = 0; i < st->targetCore - 1; i++)
			task_wait(st, i);
	}
	else
	{
		for (pos = 0; pos < rangeM; pos += HOPSIZE)
		{
			for (i = 0; i < FFTSIZE; ++i)
			{
				const float w = st->mPreWindow[i];
				st->_data[0][0][st->mBitRev[i]] = dataL[pos + i] * w;
				st->_data[1][0][st->mBitRev[i]] = dataR[pos + i] * w;
			}
			DFT4096(st->_data[0][0], st->mSineTab);
			DFT4096(st->_data[1][0], st->mSineTab);
			(*resultLRe)[idx * FFTSIZE] = st->_data[0][0][0] * 2.0f;
			(*resultRRe)[idx * FFTSIZE] = st->_data[1][0][0] * 2.0f;
			(*resultLIm)[idx * FFTSIZE] = 0.0f;
			(*resultRIm)[idx * FFTSIZE] = 0.0f;
			for (i = 1; i < HALFWNDLEN; i++)
			{
				symIdx = FFTSIZE - i;
				lR = st->_data[0][0][i] + st->_data[0][0][symIdx];
				lI = st->_data[0][0][i] - st->_data[0][0][symIdx];
				rR = st->_data[1][0][i] + st->_data[1][0][symIdx];
				rI = st->_data[1][0][i] - st->_data[1][0][symIdx];
				(*resultLRe)[idx * FFTSIZE + i] = lR;
				(*resultLIm)[idx * FFTSIZE + i] = lI;
				(*resultRRe)[idx * FFTSIZE + i] = rR;
				(*resultRIm)[idx * FFTSIZE + i] = rI;
			}
			idx++;
		}
	}
	for (i = 0; i < FFTSIZE; ++i)
	{
		const float w = st->mPreWindow[i];
		if (pos + i < data_size)
		{
			st->_data[0][0][st->mBitRev[i]] = dataL[pos + i] * w;
			st->_data[1][0][st->mBitRev[i]] = dataR[pos + i] * w;
		}
		else
		{
			st->_data[0][0][st->mBitRev[i]] = 0.0f;
			st->_data[1][0][st->mBitRev[i]] = 0.0f;
		}
	}
	DFT4096(st->_data[0][0], st->mSineTab);
	DFT4096(st->_data[1][0], st->mSineTab);
	(*resultLRe)[idx * FFTSIZE] = st->_data[0][0][0] * 2.0f;
	(*resultRRe)[idx * FFTSIZE] = st->_data[1][0][0] * 2.0f;
	(*resultLIm)[idx * FFTSIZE] = 0.0f;
	(*resultRIm)[idx * FFTSIZE] = 0.0f;
	for (i = 1; i < HALFWNDLEN; i++)
	{
		symIdx = FFTSIZE - i;
		bitRevFwd = st->mBitRev[i];
		bitRevSym = st->mBitRev[symIdx];
		lR = st->_data[0][0][i] + st->_data[0][0][symIdx];
		lI = st->_data[0][0][i] - st->_data[0][0][symIdx];
		rR = st->_data[1][0][i] + st->_data[1][0][symIdx];
		rI = st->_data[1][0][i] - st->_data[1][0][symIdx];
		(*resultLRe)[idx * FFTSIZE + i] = lR;
		(*resultLIm)[idx * FFTSIZE + i] = lI;
		(*resultRRe)[idx * FFTSIZE + i] = rR;
		(*resultRIm)[idx * FFTSIZE + i] = rI;
	}
	return result_size;
}
size_t istft(OfflineSTFT *st, float *dataLRe, float *dataLIm, float *dataRRe, float *dataRIm, size_t data_size, float **resultL, float **resultR)
{
	ISTFTThread *istftThreadData = (ISTFTThread*)st->istftThreadData;
	size_t i, j;
    size_t result_size = data_size * HOPSIZE + (FFTSIZE - HOPSIZE);
    *resultL = (float *)calloc(result_size, sizeof(float));
	*resultR = (float *)calloc(result_size, sizeof(float));
	size_t workloadMinor = data_size / st->targetCore;
	size_t symIdx;
	unsigned int bitRevFwd, bitRevSym;
	if (st->targetCore > 1)
	{
		for (i = 0; i < st->targetCore; i++)
		{
			istftThreadData[i].rangeMin = i * workloadMinor;
			istftThreadData[i].rangeMax = i * workloadMinor + workloadMinor;
			istftThreadData[i].dataLRe = dataLRe;
			istftThreadData[i].dataLIm = dataLIm;
			istftThreadData[i].dataRRe = dataRRe;
			istftThreadData[i].dataRIm = dataRIm;
		}
		istftThreadData[st->targetCore - 1].rangeMin = istftThreadData[st->targetCore - 2].rangeMax;
		istftThreadData[st->targetCore - 1].rangeMax = data_size;
		for (i = 0; i < st->targetCore - 1; i++)
			task_start(st, i + (st->targetCore - 1));
		for (i = istftThreadData[st->targetCore - 1].rangeMin; i < istftThreadData[st->targetCore - 1].rangeMax; ++i)
		{
			st->_data[0][st->targetCore - 1][0] = dataLRe[i * FFTSIZE];
			st->_data[1][st->targetCore - 1][0] = dataRRe[i * FFTSIZE];
			for (j = 1; j < HALFWNDLEN; j++)
			{
				symIdx = FFTSIZE - j;
				bitRevFwd = st->mBitRev[j];
				bitRevSym = st->mBitRev[symIdx];
				st->_data[0][st->targetCore - 1][bitRevFwd] = dataLRe[i * FFTSIZE + j] + dataLIm[i * FFTSIZE + j];
				st->_data[0][st->targetCore - 1][bitRevSym] = dataLRe[i * FFTSIZE + j] - dataLIm[i * FFTSIZE + j];
				st->_data[1][st->targetCore - 1][bitRevFwd] = dataRRe[i * FFTSIZE + j] + dataRIm[i * FFTSIZE + j];
				st->_data[1][st->targetCore - 1][bitRevSym] = dataRRe[i * FFTSIZE + j] - dataRIm[i * FFTSIZE + j];
			}
			DFT4096(st->_data[0][st->targetCore - 1], st->mSineTab);
			DFT4096(st->_data[1][st->targetCore - 1], st->mSineTab);
			memcpy(&dataLRe[i * FFTSIZE], st->_data[0][st->targetCore - 1], FFTSIZE * sizeof(float));
			memcpy(&dataRRe[i * FFTSIZE], st->_data[1][st->targetCore - 1], FFTSIZE * sizeof(float));
		}
		for (i = 0; i < st->targetCore - 1; i++)
			task_wait(st, i + (st->targetCore - 1));
		for (i = 0; i < data_size; ++i)
		{
			for (size_t pos = 0; pos < FFTSIZE; ++pos)
			{
				size_t r_pos = pos + i * HOPSIZE;
				(*resultL)[r_pos] += dataLRe[i * FFTSIZE + pos] * st->mPostWindow[pos];
				(*resultR)[r_pos] += dataRRe[i * FFTSIZE + pos] * st->mPostWindow[pos];
			}
		}
	}
	else
	{
		for (i = 0; i < data_size; ++i)
		{
			st->_data[0][0][0] = dataLRe[i * FFTSIZE];
			st->_data[1][0][0] = dataRRe[i * FFTSIZE];
			for (j = 1; j < HALFWNDLEN; j++)
			{
				symIdx = FFTSIZE - j;
				bitRevFwd = st->mBitRev[j];
				bitRevSym = st->mBitRev[symIdx];
				st->_data[0][0][bitRevFwd] = dataLRe[i * FFTSIZE + j] + dataLIm[i * FFTSIZE + j];
				st->_data[0][0][bitRevSym] = dataLRe[i * FFTSIZE + j] - dataLIm[i * FFTSIZE + j];
				st->_data[1][0][bitRevFwd] = dataRRe[i * FFTSIZE + j] + dataRIm[i * FFTSIZE + j];
				st->_data[1][0][bitRevSym] = dataRRe[i * FFTSIZE + j] - dataRIm[i * FFTSIZE + j];
			}
			DFT4096(st->_data[0][0], st->mSineTab);
			DFT4096(st->_data[1][0], st->mSineTab);
			for (size_t pos = 0; pos < FFTSIZE; ++pos)
			{
				size_t r_pos = pos + i * HOPSIZE;
				(*resultL)[r_pos] += st->_data[0][0][pos] * st->mPostWindow[pos];
				(*resultR)[r_pos] += st->_data[1][0][pos] * st->mPostWindow[pos];
			}
		}
	}
	return result_size;
}