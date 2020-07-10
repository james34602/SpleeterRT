#include <JuceHeader.h>
//#define GUI_DEBUG
#ifdef GUI_DEBUG
int isStandalone;
#endif
extern "C"
{
#include "Spleeter4Stems.h"
}
enum { PARAM_DUMMY, PARAM__MAX };
const char paramName[PARAM__MAX][2][12] =
{
{"dummy", "val" },
};
const float paramValueRange[PARAM__MAX][5] =
{
	{ 0.0f, 1.0f, 0.5f, 1.0f, 1.0f }
};
#include <userenv.h>  // GetUserProfileDirectory() (link with userenv)
const int BUFLEN = 512;
BOOL getCurrentUserDir(char* buf, DWORD buflen)
{
	HANDLE hToken;
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_READ, &hToken))
		return FALSE;
	if (!GetUserProfileDirectory(hToken, buf, &buflen))
		return FALSE;
	CloseHandle(hToken);
	return TRUE;
}
//#define DLINESUBWOOFER
class MS5_1AI : public AudioProcessor, public AudioProcessorValueTreeState::Listener
{
public:
	//==============================================================================
	MS5_1AI() : AudioProcessor(), treeState(*this, nullptr, "Parameters", createParameterLayout())
	{
#ifdef GUI_DEBUG
		isStandalone = JUCEApplicationBase::isStandaloneApp();
#endif
		int i;
		for (i = 0; i < PARAM__MAX; i++)
			treeState.addParameterListener(paramName[i][0], this);
		fs = 44100;
		char dir[512];
		getCurrentUserDir(dir, 512);
		Spleeter4StemsInit(&msr, 2048, 128, dir);
		setLatencySamples(LATENCY);
	}
	MS5_1AI::~MS5_1AI()
	{
		Spleeter4StemsFree(&msr);
	}
	AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
	{
		std::vector<std::unique_ptr<RangedAudioParameter>> parameters;
		for (int i = 0; i < PARAM__MAX; i++)
			parameters.push_back(std::make_unique<AudioParameterFloat>(paramName[i][0], paramName[i][0], NormalisableRange<float>(paramValueRange[i][0], paramValueRange[i][1]), paramValueRange[i][2]));
		return { parameters.begin(), parameters.end() };
	}
	void parameterChanged(const String& parameter, float newValue) override
	{
		if (parameter == paramName[0][0])
		{
		}
	}
	//==============================================================================
	void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override
	{
		fs = (int)sampleRate;
	}
	void releaseResources() override
	{
	}

	void processBlock(AudioBuffer<float>& buffer, MidiBuffer&) override
	{
#ifdef GUI_DEBUG
		if (isStandalone)
			return;
#endif
		// number of samples per buffer
		const int n = buffer.getNumSamples();
		// input channels
		const float *inputs[2] = { buffer.getReadPointer(0), buffer.getReadPointer(1) };
		float* const outputs[8] = { buffer.getWritePointer(0), buffer.getWritePointer(1),
		buffer.getWritePointer(2), buffer.getWritePointer(3), buffer.getWritePointer(4),
		buffer.getWritePointer(5), buffer.getWritePointer(6), buffer.getWritePointer(7) };
		// output channels
		int i, offset = 0;
		while (offset < n)
		{
			float *ptr[8] = { outputs[0] + offset, outputs[1] + offset,
			outputs[2] + offset, outputs[3] + offset, outputs[4] + offset,
			outputs[5] + offset, outputs[6] + offset, outputs[7] + offset };
			const int processing = min(n - offset, OVPSIZE);
			Spleeter4StemsProcessSamples(&msr, inputs[0] + offset, inputs[1] + offset, processing, ptr);
			offset += processing;
		}
	}

	//==============================================================================
	AudioProcessorEditor* createEditor() override { return new GenericAudioProcessorEditor(*this); }
	bool hasEditor() const override { return true; }
	const String getName() const override { return "Spleeter4Stems"; }
	bool acceptsMidi() const override { return false; }
	bool producesMidi() const override { return false; }
	double getTailLengthSeconds() const override { return 0.0; }
	int getNumPrograms() override { return 1; }
	int getCurrentProgram() override { return 0; }
	void setCurrentProgram(int) override {}
	const String getProgramName(int) override { return {}; }
	void changeProgramName(int, const String&) override {}
	bool isVST2() const noexcept { return (wrapperType == wrapperType_VST); }

	//==============================================================================
	inline double map(double x, double in_min, double in_max, double out_min, double out_max)
	{
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	}
	void getStateInformation(MemoryBlock& destData) override
		{
			MemoryOutputStream stream(destData, true);
			for (int i = 0; i < PARAM__MAX; i++)
				stream.writeFloat(treeState.getParameter(paramName[i][0])->getValue());
		}
		void setStateInformation(const void* data, int sizeInBytes) override
		{
			MemoryInputStream stream(data, static_cast<size_t> (sizeInBytes), false);
			for (int i = 0; i < PARAM__MAX; i++)
				treeState.getParameter(paramName[i][0])->setValueNotifyingHost(stream.readFloat());
		}
private:
	int isStandalone;
	//==============================================================================
	Spleeter4Stems msr;
	int fs;
	AudioProcessorValueTreeState treeState;
	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MS5_1AI)
};
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
	return new MS5_1AI();
}
