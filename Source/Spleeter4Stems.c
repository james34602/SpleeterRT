#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "codelet.h"
#include "Spleeter4Stems.h"
unsigned int LLIntegerLog2(unsigned int v)
{
	unsigned int i = 0;
	while (v > 1)
	{
		++i;
		v >>= 1;
	}
	return i;
}
unsigned LLRevBits(unsigned int x, unsigned int bits)
{
	unsigned int y = 0;
	while (bits--)
	{
		y = (y + y) + (x & 1);
		x >>= 1;
	}
	return y;
}
void LLbitReversalTbl(unsigned dst[FFTSIZE])
{
	unsigned int bits = LLIntegerLog2(FFTSIZE);
	for (int i = 0; i < FFTSIZE; ++i)
		dst[i] = LLRevBits(i, bits);
}
void LLsinHalfTblFloat(float dst[FFTSIZE])
{
	const double twopi_over_n = 6.283185307179586476925286766559 / FFTSIZE;
	for (int i = 0; i < FFTSIZE; ++i)
		dst[i] = (float)sin(twopi_over_n * i);
}
void *task_type1(void *arg)
{
	pt_infoSpleeter4s *info = (pt_infoSpleeter4s *)arg;
	Spleeter4Stems *msr = (Spleeter4Stems*)info->parentStruct;
	int i, symIdx, bitRevFwd, bitRevSym;
	float mask3L, mask3R, mask4L, mask4R;
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
		msr->timeDomainOut[4][0] = msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][2][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
		msr->timeDomainOut[5][0] = msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][2][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
		msr->timeDomainOut[6][0] = msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][3][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
		msr->timeDomainOut[7][0] = msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][3][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
		for (i = 1; i < HALFWNDLEN; i++)
		{
			symIdx = FFTSIZE - i;
			bitRevFwd = msr->mBitRev[i];
			bitRevSym = msr->mBitRev[symIdx];
			mask3L = 0.25f, mask3R = 0.25f, mask4L = 0.25f, mask4R = 0.25f;
			if (i < msr->analyseBinLimit)
			{
				mask3L = msr->maskPtr[msr->outputFramePtr][2][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
				mask3R = msr->maskPtr[msr->outputFramePtr][2][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
				mask4L = msr->maskPtr[msr->outputFramePtr][3][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
				mask4R = msr->maskPtr[msr->outputFramePtr][3][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
			}
			msr->timeDomainOut[4][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask3L;
			msr->timeDomainOut[4][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask3L;
			msr->timeDomainOut[5][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask3R;
			msr->timeDomainOut[5][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask3R;
			msr->timeDomainOut[6][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask4L;
			msr->timeDomainOut[6][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask4L;
			msr->timeDomainOut[7][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask4R;
			msr->timeDomainOut[7][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask4R;
		}
		DFT4096(msr->mTempRBuffer, msr->mSineTab);
		DFT4096(msr->timeDomainOut[4], msr->mSineTab);
		DFT4096(msr->timeDomainOut[5], msr->mSineTab);
		DFT4096(msr->timeDomainOut[6], msr->mSineTab);
		DFT4096(msr->timeDomainOut[7], msr->mSineTab);
		for (i = 0; i < FFTSIZE - SAMPLESHIFT; i++)
		{
			msr->timeDomainOut[4][i] = msr->timeDomainOut[4][i + SAMPLESHIFT] * msr->synthesisWnd[i];
			msr->timeDomainOut[5][i] = msr->timeDomainOut[5][i + SAMPLESHIFT] * msr->synthesisWnd[i];
			msr->timeDomainOut[6][i] = msr->timeDomainOut[6][i + SAMPLESHIFT] * msr->synthesisWnd[i];
			msr->timeDomainOut[7][i] = msr->timeDomainOut[7][i + SAMPLESHIFT] * msr->synthesisWnd[i];
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
void *task_type2(void *arg)
{
	pt_infoSpleeter4s *info = (pt_infoSpleeter4s *)arg;
	Spleeter4Stems *msr = (Spleeter4Stems*)info->parentStruct;
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
		processSpleeter(msr->nn[info->pos], msr->tmpSpectro, msr->maskPtr[!msr->outputFramePtr][info->pos]);
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
inline void task_start(Spleeter4Stems *PMA, int task)
{
	pt_infoSpleeter4s *info = &(PMA->shared_info[task]);
	// ensure worker is waiting
	pthread_mutex_lock(&(info->work_mtx));
	// set job information & state
	info->state = WORKING;
	// wake-up signal
	pthread_cond_signal(&(info->work_cond));
	pthread_mutex_unlock(&(info->work_mtx));
}
inline void task_wait(Spleeter4Stems *PMA, int task)
{
	pt_infoSpleeter4s *info = &(PMA->shared_info[task]);
	while (1)
	{
		pthread_cond_wait(&(info->boss_cond), &(info->boss_mtx));
		if (IDLE == info->state)
			break;
	}
}
void thread_initSpleeter4s(Spleeter4Stems *PMA)
{
	int i;
	pt_infoSpleeter4s *info = NULL;
	for (i = 0; i < TASK_NB; i++)
	{
		//		GFT->sa_log_d(TAG, "Thread %d initialized", i + 1);
		info = &(PMA->shared_info[i]);
		info->state = SETUP;
		pthread_cond_init(&(info->work_cond), NULL);
		pthread_mutex_init(&(info->work_mtx), NULL);
		pthread_cond_init(&(info->boss_cond), NULL);
		pthread_mutex_init(&(info->boss_mtx), NULL);
		pthread_mutex_lock(&(info->boss_mtx));
		info->parentStruct = (void*)PMA;
		if (!i)
		{
			pthread_create(&PMA->threads[i], NULL, task_type1, (void *)info);
			task_wait(PMA, i);
		}
		else
		{
			info->pos = i - 1;
			pthread_create(&PMA->threads[i], NULL, task_type2, (void *)info);
		}
	}
}
void thread_exitSpleeter4s(Spleeter4Stems *PMA)
{
	pt_infoSpleeter4s *info = NULL;
	for (int i = 0; i < TASK_NB; i++)
	{
		//		GFT->sa_log_d(TAG, "Thread %d get off from work", i + 1);
		info = &(PMA->shared_info[i]);
		// ensure the worker is waiting
		pthread_mutex_lock(&(info->work_mtx));
		info->state = GET_OFF_FROM_WORK;
		// wake-up signal
		pthread_cond_signal(&(info->work_cond));
		pthread_mutex_unlock(&(info->work_mtx));
		// wait for thread to exit
		pthread_join(PMA->threads[i], NULL);
		pthread_mutex_destroy(&(info->work_mtx));
		pthread_cond_destroy(&(info->work_cond));
		pthread_mutex_unlock(&(info->boss_mtx));
		pthread_mutex_destroy(&(info->boss_mtx));
		pthread_cond_destroy(&(info->boss_cond));
	}
}
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
void writeNChannelsImg2Folder(float *x, int ch, int w, int h, char *foldername, float linGain)
{
	int nError = 0;
#if defined(_WIN32)
	nError = _mkdir(foldername); // can be used on Windows
#else
	mode_t nMode = 0733; // UNIX style permissions
	nError = mkdir(foldername, nMode); // can be used on non-Windows
#endif
	int s, i;
	int whProd = w * h;
	unsigned char *y = (unsigned char*)malloc(whProd * sizeof(unsigned char));
	size_t currentMaxCharLen = 4;
	char *filename = (char*)malloc(currentMaxCharLen * sizeof(char));
	for (s = 0; s < ch; s++)
	{
		for (i = 0; i < w * h; i++)
		{
			float val = fabsf(x[s * whProd + i]) * linGain;
			if (val > 255.0f)
				val = 255.0f;
			if (val < 0.0f)
				val = 0.0f;
			y[i] = (unsigned char)val;
		}
		size_t bufsz = snprintf(NULL, 0, "%s_%d.png", foldername, s);
		if ((bufsz + 1) > currentMaxCharLen)
		{
			currentMaxCharLen = bufsz + 1;
			filename = (char*)realloc(filename, currentMaxCharLen);
		}
		snprintf(filename, bufsz + 1, "%s/%d.png", foldername, s);
		stbi_write_png(filename, w, h, 1, y, w);
	}
	free(filename);
	free(y);
}
void LLPAMSProcessNPR(Spleeter4Stems *msr)
{
	int i, j;
	// copy to temporary buffer and FHT
	for (i = 0; i < FFTSIZE; ++i)
	{
		const unsigned int k = (i + msr->mInputPos) & MINUSFFTSIZE;
		const float w = msr->analysisWnd[i];
		msr->mTempLBuffer[msr->mBitRev[i]] = msr->mInput[0][k] * w;
		msr->mTempRBuffer[msr->mBitRev[i]] = msr->mInput[1][k] * w;
	}
	int symIdx;
	unsigned int bitRevFwd, bitRevSym;
	float mask1L, mask1R, mask2L, mask2R;
	task_start(msr, 0);
	msr->timeDomainOut[0][0] = msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][0][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
	msr->timeDomainOut[1][0] = msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][0][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
	msr->timeDomainOut[2][0] = msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][1][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
	msr->timeDomainOut[3][0] = msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor] * msr->maskPtr[msr->outputFramePtr][1][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0];
	for (i = 1; i < HALFWNDLEN; i++)
	{
		symIdx = FFTSIZE - i;
		bitRevFwd = msr->mBitRev[i];
		bitRevSym = msr->mBitRev[symIdx];
		mask1L = 0.25f, mask1R = 0.25f, mask2L = 0.0f, mask2R = 0.0f;
		if (i < msr->analyseBinLimit)
		{
			mask1L = msr->maskPtr[msr->outputFramePtr][0][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
			mask1R = msr->maskPtr[msr->outputFramePtr][0][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
			mask2L = msr->maskPtr[msr->outputFramePtr][1][0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
			mask2R = msr->maskPtr[msr->outputFramePtr][1][1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i];
		}
		msr->timeDomainOut[0][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask1L;
		msr->timeDomainOut[0][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask1L;
		msr->timeDomainOut[1][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask1R;
		msr->timeDomainOut[1][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask1R;
		msr->timeDomainOut[2][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask2L;
		msr->timeDomainOut[2][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i]) * mask2L;
		msr->timeDomainOut[3][bitRevFwd] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] + msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask2R;
		msr->timeDomainOut[3][bitRevSym] = (msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] - msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i]) * mask2R;
	}
	DFT4096(msr->mTempLBuffer, msr->mSineTab);
	DFT4096(msr->timeDomainOut[0], msr->mSineTab);
	DFT4096(msr->timeDomainOut[1], msr->mSineTab);
	DFT4096(msr->timeDomainOut[2], msr->mSineTab);
	DFT4096(msr->timeDomainOut[3], msr->mSineTab);
	for (i = 0; i < FFTSIZE - SAMPLESHIFT; i++)
	{
		msr->timeDomainOut[0][i] = msr->timeDomainOut[0][i + SAMPLESHIFT] * msr->synthesisWnd[i];
		msr->timeDomainOut[1][i] = msr->timeDomainOut[1][i + SAMPLESHIFT] * msr->synthesisWnd[i];
		msr->timeDomainOut[2][i] = msr->timeDomainOut[2][i + SAMPLESHIFT] * msr->synthesisWnd[i];
		msr->timeDomainOut[3][i] = msr->timeDomainOut[3][i + SAMPLESHIFT] * msr->synthesisWnd[i];
	}
	task_wait(msr, 0);
	msr->mOutputBufferCount++;
	float *outBuffer = msr->mOutputBuffer[msr->mOutputBufferCount - 1];
	for (i = 0; i < OUTPUTSEG; ++i)
	{
		for (j = 0; j < COMPONENTS; j++)
			outBuffer[j] = msr->mOverlapStage2dash[j][i] + msr->timeDomainOut[j][i];
		outBuffer += COMPONENTS;
		for (j = 0; j < COMPONENTS; j++)
			msr->mOverlapStage2dash[j][i] = msr->timeDomainOut[j][OUTPUTSEG + i];
	}
	// Spectral analysis
	float lR = msr->mTempLBuffer[0] * 2.0f;
	float rR = msr->mTempRBuffer[0] * 2.0f;
	float lI, rI;
	float magnitudeLeft = fabsf(lR);
	float magnitudeRight = fabsf(rR);
	msr->magnitudeSpectrogram[0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0] = magnitudeLeft * (float)FFTSIZE;
	msr->magnitudeSpectrogram[1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + 0] = magnitudeRight * (float)FFTSIZE;
	msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor] = lR;
	msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor] = rR;
	for (i = 1; i < HALFWNDLEN; i++)
	{
		symIdx = FFTSIZE - i;
		bitRevFwd = msr->mBitRev[i];
		bitRevSym = msr->mBitRev[symIdx];
		lR = msr->mTempLBuffer[i] + msr->mTempLBuffer[symIdx];
		lI = msr->mTempLBuffer[i] - msr->mTempLBuffer[symIdx];
		rR = msr->mTempRBuffer[i] + msr->mTempRBuffer[symIdx];
		rI = msr->mTempRBuffer[i] - msr->mTempRBuffer[symIdx];
		if (i < msr->analyseBinLimit)
		{
			msr->magnitudeSpectrogram[0 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i] = hypotf(lR, lI) * (float)FFTSIZE;
			msr->magnitudeSpectrogram[1 * (msr->analyseBinLimit * msr->timeStep) + msr->analyseBinLimit * msr->nnMaskCursor + i] = hypotf(rR, rI) * (float)FFTSIZE;
		}
		msr->complexSpectrogram[msr->outputFramePtr][0][HALFWNDLEN * msr->nnMaskCursor + i] = lR;
		msr->complexSpectrogram[msr->outputFramePtr][1][HALFWNDLEN * msr->nnMaskCursor + i] = lI;
		msr->complexSpectrogram[msr->outputFramePtr][2][HALFWNDLEN * msr->nnMaskCursor + i] = rR;
		msr->complexSpectrogram[msr->outputFramePtr][3][HALFWNDLEN * msr->nnMaskCursor + i] = rI;
	}
	msr->nnMaskCursor++;
	if (msr->nnMaskCursor >= msr->timeStep)
	{
		// Join mask functions from threads
		task_wait(msr, 1);
		task_wait(msr, 2);
		task_wait(msr, 3);
		task_wait(msr, 4);
		// Switch buffer
		msr->outputFramePtr = !msr->outputFramePtr;
		// Prevent race condition
		memcpy(msr->tmpSpectro, msr->magnitudeSpectrogram, msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
		// Do heavy lifting
		task_start(msr, 1);
		task_start(msr, 2);
		task_start(msr, 3);
		task_start(msr, 4);
		msr->nnMaskCursor = 0;
		//writeNChannelsImg2Folder(msr->magnitudeSpectrogram, 2, msr->analyseBinLimit, msr->timeStep, "inputSpectrogram", 10.0f);
		//writeNChannelsImg2Folder(nn->mask[0], 2, analyseBinLimit, timeStep, "drum", 100.0f);
		//writeNChannelsImg2Folder(nn->mask[1], 2, analyseBinLimit, timeStep, "bass", 100.0f);
		//writeNChannelsImg2Folder(nn->mask[2], 2, analyseBinLimit, timeStep, "accompaniment", 100.0f);
		//writeNChannelsImg2Folder(nn->mask[3], 2, analyseBinLimit, timeStep, "vocal", 100.0f);
	}
	msr->mInputSamplesNeeded = OUTPUTSEG;
}
#ifndef M_PI
#define M_PI 3.141592653589793
#endif
void getAsymmetricWindow(float *analysisWnd, float *synthesisWnd, int k, int m, double freq_temporal)
{
	int i;
	memset(synthesisWnd, 0, k * sizeof(float));
	int n = ((k - m) << 1) + 2;
	for (i = 0; i < k - m; ++i)
		analysisWnd[i] = (float)pow(0.5 * (1.0 - cos(2.0 * M_PI * (i + 1.0) / (double)n)), freq_temporal);
	n = (m << 1) + 2;
	if (freq_temporal > 2.0)
		freq_temporal = 2.0;
	for (i = k - m; i < FFTSIZE; ++i)
		analysisWnd[i] = (float)pow(sqrt(0.5 * (1.0 - cos(2.0 * M_PI * ((m + i - (k - m)) + 1.0) / (double)n))), freq_temporal);
	n = m << 1;
	for (i = k - (m << 1); i < FFTSIZE; ++i)
		synthesisWnd[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)(i - (k - (m << 1))) / (double)n))) / analysisWnd[i];
	// Pre-shift window function
	for (i = 0; i < FFTSIZE - SAMPLESHIFT; i++)
		synthesisWnd[i] = synthesisWnd[i + SAMPLESHIFT];
}
void Spleeter4StemsInit(Spleeter4Stems *msr, int initSpectralBinLimit, int initTimeStep, char *dir)
{
	memset(msr, 0, sizeof(Spleeter4Stems));
	int i;
	LLbitReversalTbl(msr->mBitRev);
	LLsinHalfTblFloat(msr->mSineTab);
	for (i = 0; i < MAX_OUTPUT_BUFFERS; i++)
		msr->mOutputBuffer[i] = msr->buffer[i];
	msr->mInputSamplesNeeded = OUTPUTSEG;
	msr->mInputPos = 0;
	msr->mOutputBufferCount = 0;
	msr->mOutputReadSampleOffset = 0;
	getAsymmetricWindow(msr->analysisWnd, msr->synthesisWnd, FFTSIZE, OVPSIZE, 1.0);
	for (i = 0; i < FFTSIZE; i++)
		msr->analysisWnd[i] *= (1.0 / FFTSIZE) * 0.5f;
	memset(msr->timeDomainOut, 0, sizeof(msr->timeDomainOut));
	msr->analyseBinLimit = initSpectralBinLimit;
	msr->timeStep = initTimeStep;
	msr->magnitudeSpectrogram = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	memset(msr->magnitudeSpectrogram, 0, msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->tmpSpectro = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->complexSpectrogram[0][0] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[0][1] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[0][2] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[0][3] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[1][0] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[1][1] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[1][2] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	msr->complexSpectrogram[1][3] = (float*)malloc(HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[0][0], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[0][1], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[0][2], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[0][3], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[1][0], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[1][1], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[1][2], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	memset(msr->complexSpectrogram[1][3], 0, HALFWNDLEN * msr->timeStep * sizeof(float));
	// Neural network
	msr->nn[0] = (spleeter)allocateSpleeterStr();
	msr->nn[1] = (spleeter)allocateSpleeterStr();
	msr->nn[2] = (spleeter)allocateSpleeterStr();
	msr->nn[3] = (spleeter)allocateSpleeterStr();
	initSpleeter(msr->nn[0], msr->analyseBinLimit, msr->timeStep, 1);
	initSpleeter(msr->nn[1], msr->analyseBinLimit, msr->timeStep, 1);
	initSpleeter(msr->nn[2], msr->analyseBinLimit, msr->timeStep, 1);
	initSpleeter(msr->nn[3], msr->analyseBinLimit, msr->timeStep, 1);
	void *ceoffProvPtr[4];
	getCoeffPtr(msr->nn[0], &ceoffProvPtr[0]);
	getCoeffPtr(msr->nn[1], &ceoffProvPtr[1]);
	getCoeffPtr(msr->nn[2], &ceoffProvPtr[2]);
	getCoeffPtr(msr->nn[3], &ceoffProvPtr[3]);
	msr->maskPtr[0][0] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[0][1] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[0][2] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[0][3] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[1][0] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[1][1] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[1][2] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	msr->maskPtr[1][3] = (float*)malloc(msr->analyseBinLimit * msr->timeStep * 2 * sizeof(float));
	for (i = 0; i < msr->analyseBinLimit * msr->timeStep * 2; i++)
	{
		float v = 1.0f;
		msr->maskPtr[0][0][i] = v;
		msr->maskPtr[0][1][i] = v;
		msr->maskPtr[0][2][i] = v;
		msr->maskPtr[0][3][i] = v;
		msr->maskPtr[1][0][i] = v;
		msr->maskPtr[1][1][i] = v;
		msr->maskPtr[1][2][i] = v;
		msr->maskPtr[1][3][i] = v;
	}
	// Load coeff
	char file1[17] = "\\drum4stems.dat";
	char file2[17] = "\\bass4stems.dat";
	char file3[26] = "\\accompaniment4stems.dat";
	char file4[17] = "\\vocal4stems.dat";
	size_t len1 = strlen(dir);
	size_t len2 = strlen(file1);
	char *concat = (char*)malloc(len1 + len2 + 1);
	memcpy(concat, dir, len1);
	memcpy(concat + len1, file1, len2 + 1);
	FILE *fp = fopen(concat, "rb");
	fread(ceoffProvPtr[0], 1, 39290900, fp);
	fclose(fp);
	free(concat);
	len2 = strlen(file2);
	concat = (char*)malloc(len1 + len2 + 1);
	memcpy(concat, dir, len1);
	memcpy(concat + len1, file2, len2 + 1);
	fp = fopen(concat, "rb");
	fread(ceoffProvPtr[1], 1, 39290900, fp);
	fclose(fp);
	free(concat);
	len2 = strlen(file3);
	concat = (char*)malloc(len1 + len2 + 1);
	memcpy(concat, dir, len1);
	memcpy(concat + len1, file3, len2 + 1);
	fp = fopen(concat, "rb");
	fread(ceoffProvPtr[2], 1, 39290900, fp);
	fclose(fp);
	free(concat);
	len2 = strlen(file4);
	concat = (char*)malloc(len1 + len2 + 1);
	memcpy(concat, dir, len1);
	memcpy(concat + len1, file4, len2 + 1);
	fp = fopen(concat, "rb");
	fread(ceoffProvPtr[3], 1, 39290900, fp);
	fclose(fp);
	free(concat);
	msr->nnMaskCursor = 0;
	msr->outputFramePtr = 0;
	thread_initSpleeter4s(msr);
}
void Spleeter4StemsFree(Spleeter4Stems *msr)
{
	if (!msr->mInputPos && !msr->nnMaskCursor && !msr->mOutputReadSampleOffset && !msr->mOutputBufferCount)
	{
		// Indicate we haven't run wake up the neural network thread for the first time
		for (int i = 1; i < TASK_NB; i++)
			task_wait(msr, i);
	}
	else
	{
		for (int i = 0; i < TASK_NB; i++)
		{
			if (msr->shared_info[i].state == WORKING)
				task_wait(msr, i);
		}
	}
	thread_exitSpleeter4s(msr);
	free(msr->maskPtr[0][0]);
	free(msr->maskPtr[0][1]);
	free(msr->maskPtr[0][2]);
	free(msr->maskPtr[0][3]);
	free(msr->maskPtr[1][0]);
	free(msr->maskPtr[1][1]);
	free(msr->maskPtr[1][2]);
	free(msr->maskPtr[1][3]);
	free(msr->nn[0]);
	free(msr->nn[1]);
	free(msr->nn[2]);
	free(msr->nn[3]);
	free(msr->tmpSpectro);
	free(msr->magnitudeSpectrogram);
	free(msr->complexSpectrogram[0][0]);
	free(msr->complexSpectrogram[0][1]);
	free(msr->complexSpectrogram[0][2]);
	free(msr->complexSpectrogram[0][3]);
	free(msr->complexSpectrogram[1][0]);
	free(msr->complexSpectrogram[1][1]);
	free(msr->complexSpectrogram[1][2]);
	free(msr->complexSpectrogram[1][3]);
}
void Spleeter4StemsProcessSamples(Spleeter4Stems *msr, const float *inLeft, const float *inRight, int inSampleCount, float **components)
{
	int i, outSampleCount, maxOutSampleCount, copyCount;
	outSampleCount = 0;
	maxOutSampleCount = inSampleCount;

	while (inSampleCount > 0) {
		copyCount = min((int)msr->mInputSamplesNeeded, inSampleCount);
		float *sampDL = &msr->mInput[0][msr->mInputPos];
		float *sampDR = &msr->mInput[1][msr->mInputPos];
		const float *max = inLeft + copyCount;
		while (inLeft < max)
		{
			*sampDL = *inLeft;
			*sampDR = *inRight;
			inLeft += 1;
			inRight += 1;
			sampDL += 1;
			sampDR += 1;
		}
		inSampleCount -= copyCount;
		msr->mInputPos = (msr->mInputPos + copyCount) & MINUSFFTSIZE;
		msr->mInputSamplesNeeded -= copyCount;
		if (msr->mInputSamplesNeeded == 0)
			LLPAMSProcessNPR(msr);
	}
	for (i = 0; i < COMPONENTS; i++)
		msr->signalIOPointer[i] = components[i];
	while ((msr->mOutputBufferCount > 0) && (outSampleCount < maxOutSampleCount)) {
		float *sampD = msr->mOutputBuffer[0];
		copyCount = min(OUTPUTSEG - msr->mOutputReadSampleOffset, maxOutSampleCount - outSampleCount);
		float *out = sampD + (msr->mOutputReadSampleOffset * COMPONENTS);
		float *max = msr->signalIOPointer[0] + copyCount;
		while (msr->signalIOPointer[0] < max)
		{
			*msr->signalIOPointer[0] = (float)*out;
			out++;
			msr->signalIOPointer[0]++;
			for (int j = 1; j < COMPONENTS; j++)
			{
				*msr->signalIOPointer[j] = (float)*out;
				out++;
				msr->signalIOPointer[j]++;
			}
		}
		outSampleCount += copyCount;
		msr->mOutputReadSampleOffset += copyCount;
		if (msr->mOutputReadSampleOffset == OUTPUTSEG)
		{
			msr->mOutputBufferCount--;
			msr->mOutputReadSampleOffset = 0;
			if (msr->mOutputBufferCount > 0) {
				int i;
				float *moveToEnd = msr->mOutputBuffer[0];
				// Shift the buffers so that the current one for reading is at index 0
				for (i = 1; i < MAX_OUTPUT_BUFFERS; i++)
					msr->mOutputBuffer[i - 1] = msr->mOutputBuffer[i];
				msr->mOutputBuffer[MAX_OUTPUT_BUFFERS - 1] = 0;
				// Move the previous first buffer to the end (first null pointer)
				for (i = 0; i < MAX_OUTPUT_BUFFERS; i++)
				{
					if (!msr->mOutputBuffer[i])
					{
						msr->mOutputBuffer[i] = moveToEnd;
						break;
					}
				}
			}
		}
	}
}