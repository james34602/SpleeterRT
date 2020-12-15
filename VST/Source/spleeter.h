typedef struct _spleeter *spleeter;
size_t getCoeffSize();
void* allocateSpleeterStr();
void initSpleeter(struct _spleeter *nn, int width, int height, int stemMode, void *coeff);
void getMaskPtr(struct _spleeter *nn, float **mask);
void freeSpleeter(struct _spleeter *nn);
void processSpleeter(struct _spleeter *nn, float *x, float *y);