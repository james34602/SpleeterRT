#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
//#include <vld.h>
#include "spleeter.h"
#include "stftFix.h"
#include "cpthread.h"
#define DRMP3_IMPLEMENTATION
#include "dr_mp3.h"
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
//  Windows
#ifdef _WIN32
#include <Windows.h>
////////////////////////////////////////////////////////////////////
// Performance timer
double get_wall_time()
{
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq))
		return 0;
	if (!QueryPerformanceCounter(&time))
		return 0;
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time()
{
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0)
		return (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	else
		return 0;
}
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL))
		return 0;
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time()
{
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif
void channel_joinFloat(float **chan_buffers, unsigned int num_channels, float *buffer, size_t num_frames, unsigned int preshift)
{
	size_t i, samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		buffer[i] = chan_buffers[i % num_channels][preshift + i / num_channels];
}
void channel_join(double **chan_buffers, unsigned int num_channels, double *buffer, unsigned int num_frames)
{
	unsigned int i, samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		buffer[i] = chan_buffers[i % num_channels][i / num_channels];
}
void channel_split(double *buffer, unsigned int num_frames, double **chan_buffers, unsigned int num_channels)
{
	unsigned int i, samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		chan_buffers[i % num_channels][i / num_channels] = buffer[i];
}
void channel_splitFloat(float *buffer, unsigned int num_frames, float **chan_buffers, unsigned int num_channels, unsigned int preshift)
{
	unsigned int i, samples = num_frames * num_channels;
	for (i = 0; i < samples; i++)
		chan_buffers[i % num_channels][preshift + i / num_channels] = buffer[i];
}
void normalise(float *buffer, int num_samps)
{
	int i;
	float max = 0.0f;
	for (i = 0; i < num_samps; i++)
		max = fabsf(buffer[i]) < max ? max : fabsf(buffer[i]);
	max = 1.0f / max;
	for (i = 0; i < num_samps; i++)
		buffer[i] *= max;
}
int get_float(char *val, float *F)
{
	char *eptr;
	errno = 0;
	float f = strtof(val, &eptr);
	if (eptr != val && errno != ERANGE)
	{
		*F = f;
		return 1;
	}
	return 0;
}
char *ft_strdup(char *src)
{
    char *str;
    char *p;
    int len = 0;

    while (src[len])
        len++;
    str = malloc(len + 1);
    p = str;
    while (*src)
        *p++ = *src++;
    *p = '\0';
    return str;
}
char *basenameM(char *path)
{
#ifdef _MSC_VER
	char *s = strrchr(path, '\\');
#else
	char *s = strrchr(path, '/');
#endif
	if (!s)
		return ft_strdup(path);
	else
		return ft_strdup(s + 1);
}
const char *get_filename_ext(const char *filename)
{
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}
static const double compressedCoeffMQ[701] = { 0.919063234986138511, 0.913619994199411201, 0.897406560667438402, 0.870768836078722797, 0.834273523109754001, 0.788693711602254766, 0.734989263333015286, 0.674282539592362951, 0.607830143521649657, 0.536991457245508341, 0.463194839157173466, 0.387902406850539450, 0.312574364478514499, 0.238633838900749129, 0.167433166845669668, 0.100222526262645231, 0.038121730693097225, -0.017904091793426027, -0.067064330735278010, -0.108757765594775291, -0.142582153044975485, -0.168338357500518510, -0.186029009837402531, -0.195851834439330574, -0.198187933840591440, -0.193585458951914369, -0.182739217157973949, -0.166466876941489483, -0.145682513177707279, -0.121368299577954988, -0.094545192393702127, -0.066243461643369750, -0.037473912797399318, -0.009200603832691301, 0.017684198632122932, 0.042382162574711987, 0.064207571946511041, 0.082602100079150684, 0.097145355203635028, 0.107561011406507381, 0.113718474453852247, 0.115630166683615837, 0.113444644504459249, 0.107435882084395071, 0.097989162055837783, 0.085584105452548076, 0.070775446120293309, 0.054172207614333223, 0.036415971885660689, 0.018158938330246815, 0.000042459196338912, -0.017323296238410713, -0.033377949403731559, -0.047627263300716267, -0.059656793081079629, -0.069142490141500798, -0.075858019256578396, -0.079678658979652428, -0.080581767609623323, -0.078643906863599317, -0.074034819562284179, -0.067008552930955145, -0.057892102695980191, -0.047072022601671190, -0.034979497375161483, -0.022074413174279148, -0.008828977400685476, 0.004288560713652965, 0.016829555699174666, 0.028379479813981756, 0.038570858162652835, 0.047094241683417769, 0.053706909605020871, 0.058239076113395197, 0.060597464446319166, 0.060766202425508065, 0.058805083502616720, 0.054845323789501445, 0.049083025498695095, 0.041770628243703020, 0.033206689594802247, 0.023724383421121997, 0.013679137601712221, 0.003435850879130631, -0.006643868309165797, -0.016214010012738603, -0.024955612937682017, -0.032586990530198853, -0.038872417226545809, -0.043629018879643239, -0.046731678346964158, -0.048115839004410917, -0.047778163016171563, -0.045775074997070689, -0.042219292770236193, -0.037274512912134725, -0.031148477572630149, -0.024084698830208532, -0.016353156105483747, -0.008240309809337577, -0.000038789761018515, 0.007962880277736915, 0.015490170167632772, 0.022291712611484882, 0.028147455724642705, 0.032875542265754551, 0.036337708303315903, 0.038443049556812471, 0.039150063091460026, 0.038466933347880143, 0.036450092517807633, 0.033201143952433128, 0.028862291652591764, 0.023610467168487866, 0.017650385903971395, 0.011206796641134264, 0.004516210167813600, -0.002181595351151269, -0.008651993358287469, -0.014673562407359826, -0.020045503214184583, -0.024594176093158650, -0.028178551235573571, -0.030694406321622035, -0.032077152831841031, -0.032303222419993387, -0.031389996039150381, -0.029394309376722470, -0.026409616804964359, -0.022561940854659814, -0.018004773700230153, -0.012913130046223239, -0.007476976122050557, -0.001894276500050309, 0.003636091270827173, 0.008921304789011335, 0.013781207467236415, 0.018054338893886482, 0.021603186795815136, 0.024318493648450956, 0.026122487293166251, 0.026970945047679402, 0.026854043263213976, 0.025795987570079431, 0.023853461640206807, 0.021112972713804853, 0.017687209024348168, 0.013710556397414673, 0.009333947668859192, 0.004719238361448204, 0.000033314715823999, -0.004557854609777880, -0.008895014112733140, -0.012831064739959125, -0.016235971633599879, -0.019000975419769615, -0.021041973670496809, -0.022301970824562707, -0.022752529440111541, -0.022394191906072568, -0.021255878361951062, -0.019393302279828196, -0.016886478718542881, -0.013836430536684995, -0.010361223853106050, -0.006591484944463394, -0.002665565936317155, 0.001275464342459697, 0.005092825309417521, 0.008654850008311749, 0.011841465274590917, 0.014548176385701671, 0.016689426206986688, 0.018201223316688792, 0.019042961289876651, 0.019198381208357294, 0.018675660452129358, 0.017506641810096972, 0.015745246837337051, 0.013465145155917528, 0.010756776112709417, 0.007723840062309154, 0.004479392878420811, 0.001141688626311485, -0.002170078649346380, -0.005339982462341837, -0.008259373919338373, -0.010830557217282604, -0.012970007990380254, -0.014611030342508765, -0.015705770153788771, -0.016226526478563909, -0.016166328657139784, -0.015538773207899238, -0.014377140707818779, -0.012732837794565421, -0.010673232279050818, -0.008278969379415602, -0.005640873626063710, -0.002856553527664500, -0.000026834265658038, 0.002747852704083721, 0.005370995258360709, 0.007753319014958258, 0.009815785813909824, 0.011492173003678219, 0.012731150958433296, 0.013497795530424229, 0.013774493273929109, 0.013561219484126362, 0.012875191572278549, 0.011749922250546383, 0.010233717662138146, 0.008387684262046795, 0.006283324310867826, 0.003999812773708582, 0.001621057822818793, -0.000767347236997687, -0.003081171091507831, -0.005240483245491547, -0.007172383140908554, -0.008813427537033921, -0.010111676574189758, -0.011028293954650051, -0.011538653669060464, -0.011632924035784708, -0.011316118830412747, -0.010607624279109693, -0.009540229002217340, -0.008158700990423423, -0.006517970800651516, -0.004680992876389242, -0.002716366825287035, -0.000695807329823454, 0.001308445056270027, 0.003226179724201707, 0.004991648959431359, 0.006545794666321473, 0.007838194454033614, 0.008828664063258869, 0.009488466505815606, 0.009801093167340614, 0.009762597931815446, 0.009381481548192093, 0.008678139395534967, 0.007683900954968532, 0.006439703144240938, 0.004994451750326167, 0.003403135115410580, 0.001724761684323746, 0.000020197792912962, -0.001650015947868542, -0.003227792151864713, -0.004659494079420105, -0.005897735119564774, -0.006902920847777659, -0.007644484483944344, -0.008101778413755888, -0.008264597354555087, -0.008133322264400958, -0.007718687720230902, -0.007041188742854554, -0.006130155461650219, -0.005022535167821778, -0.003761430832691131, -0.002394452762763960, -0.000971945496911797, 0.000454844825025831, 0.001835596641714852, 0.003122668316617104, 0.004272722114380925, 0.005248162177843576, 0.006018339594106877, 0.006560486855847911, 0.006860354383749922, 0.006912532886871314, 0.006720456790559610, 0.006296095343184246, 0.005659348921651181, 0.004837178114662079, 0.003862502033291890, 0.002772909691220670, 0.001609233980906675, 0.000414041581645497, -0.000769906024675906, -0.001901165407831948, -0.002941044990442539, -0.003854910802244932, -0.004613320774623974, -0.005192951162290093, -0.005577286818932973, -0.005757056020769449, -0.005730399990410974, -0.005502776877353867, -0.005086609356845171, -0.004500693886508907, -0.003769397706347888, -0.002921676615038254, -0.001989952181071211, -0.001008891180284735, -0.000014132577398601, 0.000958991766382216, 0.001876697269289887, 0.002707904681562794, 0.003425276863778077, 0.004006100299408528, 0.004432983601002821, 0.004694352366032693, 0.004784727410211850, 0.004704781367912914, 0.004461176612085595, 0.004066195127045020, 0.003537178100163921, 0.002895799341758980, 0.002167201988627001, 0.001379032128141500, 0.000560405874095270, -0.000259152041390146, -0.001050759947470964, -0.001787184184342184, -0.002443762818329620, -0.002999217281793143, -0.003436325772772713, -0.003742437746853627, -0.003909814939345641, -0.003935790838753386, -0.003822747153675574, -0.003577912336230492, -0.003212993416401785, -0.002743658050835511, -0.002188888608264336, -0.001570234143874397, -0.000910989133812205, -0.000235329764723561, 0.000432560641279413, 0.001069345839998360, 0.001653340745377959, 0.002165238082962834, 0.002588733711954676, 0.002911030869493576, 0.003123208352342977, 0.003220442832479460, 0.003202080909987641, 0.003071561941475896, 0.002836197950977609, 0.002506821849485185, 0.002097319592184942, 0.001624065644771860, 0.001105284094542681, 0.000560359841433201, 0.000009125484808694, -0.000528850236297424, -0.001034947274672293, -0.001492128764321782, -0.001885503916298735, -0.002202801129643487, -0.002434736758218434, -0.002575269035034838, -0.002621730990172647, -0.002574840645573092, -0.002438591168737718, -0.002220027862259672, -0.001928922712789991, -0.001577360593112676, -0.001179253996234489, -0.000749805296090004, -0.000304936916893765, 0.000139289578942291, 0.000567244609490058, 0.000964271897894877, 0.001317181565737385, 0.001614678737960774, 0.001847714105275549, 0.002009746006742852, 0.002096907004167219, 0.002108071492117578, 0.002044824491385029, 0.001911335280223238, 0.001714142804308648, 0.001461862761483392, 0.001164828784166054, 0.000834682161765550, 0.000483925998593740, 0.000125460552453497, -0.000227883269383389, -0.000563795658925073, -0.000870909262856999, -0.001139175997920627, -0.001360187135536317, -0.001527426803455576, -0.001636451679881492, -0.001684992470943874, -0.001672975660262524, -0.001602466894019970, -0.001477540114542411, -0.001304079085075351, -0.001089520173828018, -0.000842547115162114, -0.000572749884052837, -0.000290260767615715, -0.000005381173403155, 0.000271787324682131, 0.000531694720882059, 0.000765665077860444, 0.000966179147256847, 0.001127108176071712, 0.001243891947014572, 0.001313656276814221, 0.001335267482640161, 0.001309323638539731, 0.001238084697878249, 0.001125345675261167, 0.000976258990458714, 0.000797113715033597, 0.000595080778704776, 0.000377934148912837, 0.000153758569481225, -0.000069345376995143, -0.000283548328671139, -0.000481561352316614, -0.000656878246049437, -0.000803983100343503, -0.000918516742461215, -0.000997397336828164, -0.001038892182663637, -0.001042639573678904, -0.001009621396079384, -0.000942088876021349, -0.000843445486235168, -0.000718092430499951, -0.000571243299133416, -0.000408715393446694, -0.000236705827693455, -0.000061560820167541, 0.000110453421068778, 0.000273370118506192, 0.000421724138601977, 0.000550730322629083, 0.000656432310144007, 0.000735817450677950, 0.000786894740519778, 0.000808734131933592, 0.000801466989262860, 0.000766248858509478, 0.000705187025332569, 0.000621236516650982, 0.000518069214751335, 0.000399921568730665, 0.000271426983019702, 0.000137439322056229, 0.000002854088234340, -0.000127566289796932, -0.000249356967950712, -0.000358499459727905, -0.000451549869015331, -0.000525742663600061, -0.000579067066332711, -0.000610314210246149, -0.000619094306052033, -0.000605824164738963, -0.000571686465475047, -0.000518563123569660, -0.000448945962885840, -0.000365828604967228, -0.000272584032349123, -0.000172832651673869, -0.000070305865763902, 0.000031289837955523, 0.000128403456337462, 0.000217760218406138, 0.000296468531144650, 0.000362109765670523, 0.000412808249833617, 0.000447279627497663, 0.000464856578836417, 0.000465491738613502, 0.000449738470059130, 0.000418710921836804, 0.000374025488711351, 0.000317726390511799, 0.000252198560764628, 0.000180071382714710, 0.000104117018254314, 0.000027147141711448, -0.000048088182130861, -0.000118995940709898, -0.000183226726188442, -0.000238749615318076, -0.000283913107803554, -0.000317490431754438, -0.000338708143110164, -0.000347257581417331, -0.000343289373546324, -0.000327391777412823, -0.000300554208881314, -0.000264117778630104, -0.000219715066774411, -0.000169201669906033, -0.000114582260151285, -0.000057933995063949, -0.000001330110803372, 0.000053233576939992, 0.000103906741924272, 0.000149044949497856, 0.000187260510974094, 0.000217462319983746, 0.000238883649294571, 0.000251097355835207, 0.000254018413730855, 0.000247894152903089, 0.000233283008194432, 0.000211022966716673, 0.000182191226994878, 0.000148056842795767, 0.000110028310364557, 0.000069598166140896, 0.000028286691831679, -0.000012413223177511, -0.000051088131137235, -0.000086451240238396, -0.000117383287411682, -0.000142965848616867, -0.000162506184310969, -0.000175553056891817, -0.000181903303379713, -0.000181599287597338, -0.000174917679492114, -0.000162350303974188, -0.000144578058136870, -0.000122439106161612, -0.000096892719738482, -0.000068980234721197, -0.000039784640417051, -0.000010390306964679, 0.000018155708707897, 0.000044879452343343, 0.000068911789142910, 0.000089515073669816, 0.000106104020030281, 0.000118260259226232, 0.000125740319217347, 0.000128477011389261, 0.000126574445796332, 0.000120297118433995, 0.000110053709582710, 0.000096376396848920, 0.000079896615190895, 0.000061318285745045, 0.000041389584008291, 0.000020874325790195, 0.000000524017729422, -0.000018948449136533, -0.000036892585232981, -0.000052740818819573, -0.000066025144314650, -0.000076389469747209, -0.000083597404911355, -0.000087535408131387, -0.000088211381288728, -0.000085748963835814, -0.000080377921496540, -0.000072421149619506, -0.000062278911057702, -0.000050411001395778, -0.000037317578835504, -0.000023519411693731, -0.000009538283954733, 0.000004121739654397, 0.000016991550435446, 0.000028652179052461, 0.000038747470023706, 0.000046993903986995, 0.000053187343852112, 0.000057206603209622, 0.000059013855762422, 0.000058652018744224, 0.000056239347370971, 0.000051961567969763, 0.000046061951828715, 0.000038829788015541, 0.000030587750191730, 0.000021678669346899, 0.000012452221695327, 0.000003252019725938, -0.000005596443768274, -0.000013796601731427, -0.000021090036946349, -0.000027263866219030, -0.000032156106624670, -0.000035658928433894, -0.000037719781474555, -0.000038340460301533, -0.000037574245901964, -0.000035521325378456, -0.000032322744186229, -0.000028153186597604, -0.000023212908194301, -0.000017719158939021, -0.000011897436866734, -0.000005972901266780, -0.000000162251446552, 0.000005333655793588, 0.000010336218246169, 0.000014694346087313, 0.000018288433744639, 0.000021032968627464, 0.000022877753957232, 0.000023807775391314, 0.000023841789731823, 0.000023029757133319, 0.000021449274538926, 0.000019201196577477, 0.000016404650185715, 0.000013191660473528, 0.000009701607868611, 0.000006075730729441, 0.000002451874068818, -0.000001040335292299, -0.000004283732788372, -0.000007177155365100, -0.000009638185206925, -0.000011605042076524, -0.000013037603419542, -0.000013917565218600, -0.000014247788012456, -0.000014050900469913, -0.000013367256554309, -0.000012252360976939, -0.000010773890887184, -0.000009008449384796, -0.000007038188493661, -0.000004947435946560, -0.000002819451929447, -0.000000733429418017, 0.000001238164361723, 0.000003031826677860, 0.000004594780305705, 0.000005886220575305, 0.000006878033995645, 0.000007554995005623, 0.000007914466875845, 0.000007965650064675, 0.000007728435880102, 0.000007231934716903, 0.000006512756173231, 0.000005613122895949, 0.000004578901096601, 0.000003457628489583, 0.000002296615219305, 0.000001141185552533, 0.000000033118183302, -0.000000990668545558, -0.000001899152803076, -0.000002667946210802, -0.000003279724008567, -0.000003724353373815, -0.000003998736284580, -0.000004106393367485, -0.000004056823505075, -0.000003864680371659, -0.000003548811395079, -0.000003131206866384, -0.000002635907089885, -0.000002087913711897, -0.000001512147886406, -0.000000932492985725, -0.000000370953442845, 0.000000153045653294, 0.000000623201057861, 0.000001026706448750, 0.000001354458044511, 0.000001601095747237, 0.000001764891111188, 0.000001847498832724, 0.000001853592772907, 0.000001790410657756, 0.000001667233505382, 0.000001494826511430, 0.000001284867642241, 0.000001049388641653, 0.000000800250692844, 0.000000548673760673, 0.000000304834862491, 0.000000077546377014, -0.000000125978796206, -0.000000300272674786, -0.000000441669721214, -0.000000548281621807, -0.000000619897641839, -0.000000657821438861, -0.000000664657100282, -0.000000644058349094, -0.000000600455333568, -0.000000538773213853, -0.000000464155939282, -0.000000381707256690, -0.000000296259201847, -0.000000212176215381, -0.000000133200709942, -0.000000062343516559, 0.0 };
void decompressResamplerMQ(const double y[701], float *yi)
{
	double breaks[701];
	double coefs[2800];
	int k;
	double s[701];
	double dx[700];
	double dvdf[700];
	double r, dzzdx, dzdxdx;
	for (k = 0; k < 700; k++)
	{
		r = 0.0014285714285714286 * ((double)k + 1.0) - 0.0014285714285714286 * (double)k;
		dx[k] = r;
		dvdf[k] = (y[k + 1] - y[k]) / r;
	}
	s[0] = ((dx[0] + 0.0057142857142857143) * dx[1] * dvdf[0] + dx[0] * dx[0] * dvdf[1]) / 0.0028571428571428571;
	s[700] = ((dx[699] + 0.0057142857142857828) * dx[698] * dvdf[699] + dx[699] * dx[699] * dvdf[698]) / 0.0028571428571428914;
	breaks[0] = dx[1];
	breaks[700] = dx[698];
	for (k = 0; k < 699; k++)
	{
		r = dx[k + 1];
		s[k + 1] = 3.0 * (r * dvdf[k] + dx[k] * dvdf[k + 1]);
		breaks[k + 1] = 2.0 * (r + dx[k]);
	}
	r = dx[1] / breaks[0];
	breaks[1] -= r * 0.0028571428571428571;
	s[1] -= r * s[0];
	for (k = 0; k < 698; k++)
	{
		r = dx[k + 2] / breaks[k + 1];
		breaks[k + 2] -= r * dx[k];
		s[k + 2] -= r * s[k + 1];
	}
	r = 0.0028571428571428914 / breaks[699];
	breaks[700] -= r * dx[698];
	s[700] -= r * s[699];
	s[700] /= breaks[700];
	for (k = 698; k >= 0; k--)
		s[k + 1] = (s[k + 1] - dx[k] * s[k + 2]) / breaks[k + 1];
	s[0] = (s[0] - 0.0028571428571428571 * s[1]) / breaks[0];
	for (k = 0; k < 701; k++)
		breaks[k] = 0.0014285714285714286 * (double)k;
	for (k = 0; k < 700; k++)
	{
		r = 1.0 / dx[k];
		dzzdx = (dvdf[k] - s[k]) * r;
		dzdxdx = (s[k + 1] - dvdf[k]) * r;
		coefs[k] = (dzdxdx - dzzdx) * r;
		coefs[k + 700] = 2.0 * dzzdx - dzdxdx;
		coefs[k + 1400] = s[k];
		coefs[k + 2100] = y[k];
	}
	double d = 1.0 / 22437.0;
	int low_i, low_ip1, high_i, mid_i;
	for (k = 0; k < 22438; k++)
	{
		low_i = 0;
		low_ip1 = 2;
		high_i = 701;
		r = k * d;
		while (high_i > low_ip1)
		{
			mid_i = ((low_i + high_i) + 1) >> 1;
			if (r >= breaks[mid_i - 1])
			{
				low_i = mid_i - 1;
				low_ip1 = mid_i + 1;
			}
			else
				high_i = mid_i;
		}
		double xloc = r - breaks[low_i];
		yi[k] = xloc * (xloc * (xloc * coefs[low_i] + coefs[low_i + 700]) + coefs[low_i + 1400]) + coefs[low_i + 2100];
	}
}
#include "libsamplerate/samplerate.h"
void JamesDSPOfflineResampling(float const *in, float *out, size_t lenIn, size_t lenOut, int channels, double src_ratio)
{
	if (lenOut == lenIn && lenIn == 1)
	{
		memcpy(out, in, channels * sizeof(float));
		return;
	}
	SRC_DATA src_data;
	memset(&src_data, 0, sizeof(src_data));
	src_data.data_in = in;
	src_data.data_out = out;
	src_data.input_frames = lenIn;
	src_data.output_frames = lenOut;
	src_data.src_ratio = src_ratio;
	int error;
	if ((error = src_simple(&src_data, 0, channels)))
	{
		printf("\n%s\n\n", src_strerror(error));
	}
}
float* loadAudioFile(char *filename, double targetFs, unsigned int *channels, drwav_uint64 *totalPCMFrameCount)
{
	unsigned int fs = 1;
    const char *ext = get_filename_ext(filename);
    float *pSampleData = 0;
    if (!strncmp(ext, "wav", 5))
        pSampleData = drwav_open_file_and_read_pcm_frames_f32(filename, channels, &fs, totalPCMFrameCount, 0);
    if (!strncmp(ext, "flac", 5))
        pSampleData = drflac_open_file_and_read_pcm_frames_f32(filename, channels, &fs, totalPCMFrameCount, 0);
    if (!strncmp(ext, "mp3", 5))
    {
        drmp3_config mp3Conf;
        pSampleData = drmp3_open_file_and_read_pcm_frames_f32(filename, &mp3Conf, totalPCMFrameCount, 0);
        *channels = mp3Conf.channels;
        fs = mp3Conf.sampleRate;
    }
	if (pSampleData == NULL)
	{
		printf("Error opening and reading WAV file");
		return 0;
	}
	// Sanity check
	if (*channels < 1)
	{
		printf("Invalid audio channels count");
		free(pSampleData);
		return 0;
	}
	if ((*totalPCMFrameCount <= 0) || (*totalPCMFrameCount <= 0))
	{
		printf("Invalid audio sample rate / frame count");
		free(pSampleData);
		return 0;
	}
	double ratio = targetFs / (double)fs;
	if (ratio != 1.0)
	{
		int compressedLen = (int)ceil(*totalPCMFrameCount * ratio);
		float *tmpBuf = (float*)malloc(compressedLen * *channels * sizeof(float));
		memset(tmpBuf, 0, compressedLen * *channels * sizeof(float));
		JamesDSPOfflineResampling(pSampleData, tmpBuf, *totalPCMFrameCount, compressedLen, *channels, ratio);
		*totalPCMFrameCount = compressedLen;
		free(pSampleData);
		return tmpBuf;
	}
	return pSampleData;
}
float *decompressedCoefficients;
int ispowerof2(size_t x)
{
	return x && !(x & (x - 1));
}
typedef struct
{
	enum pt_state state;
	pthread_cond_t work_cond;
	pthread_mutex_t work_mtx;
	pthread_cond_t boss_cond;
	pthread_mutex_t boss_mtx;

	size_t rangeMin, rangeMax;
	size_t *analyseBinLimit, *timeStep;

	spleeter nn;
	float *unaffectedWeight, *reL, *imL, *reR, *imR, *magnitudeSpectrogram, *maskPtr;
} separationSubthread;
void *task_Separation(void *arg)
{
	separationSubthread *info = (separationSubthread *)arg;
	size_t i, nnMaskCursor;
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
		for (size_t j = info->rangeMin; j < info->rangeMax; j++)
		{
			size_t nnStep = j * *info->timeStep * FFTSIZE;
			for (nnMaskCursor = 0; nnMaskCursor < *info->timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < *info->analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					info->magnitudeSpectrogram[0 * (*info->analyseBinLimit * *info->timeStep) + *info->analyseBinLimit * nnMaskCursor + i] = hypotf(info->reL[idx], info->imL[idx]) * (float)FFTSIZE;
					info->magnitudeSpectrogram[1 * (*info->analyseBinLimit * *info->timeStep) + *info->analyseBinLimit * nnMaskCursor + i] = hypotf(info->reR[idx], info->imR[idx]) * (float)FFTSIZE;
				}
			}
			processSpleeter(info->nn, info->magnitudeSpectrogram, info->maskPtr);
			for (nnMaskCursor = 0; nnMaskCursor < *info->timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < *info->analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					float maskL = info->maskPtr[0 * (*info->analyseBinLimit * *info->timeStep) + *info->analyseBinLimit * nnMaskCursor + i];
					float maskR = info->maskPtr[1 * (*info->analyseBinLimit * *info->timeStep) + *info->analyseBinLimit * nnMaskCursor + i];
					info->reL[idx] *= maskL;
					info->imL[idx] *= maskL;
					info->reR[idx] *= maskR;
					info->imR[idx] *= maskR;
				}
				for (; i < HALFWNDLEN; i++)
				{
					size_t idx = offset + i;
					info->reL[idx] *= *info->unaffectedWeight;
					info->imL[idx] *= *info->unaffectedWeight;
					info->reR[idx] *= *info->unaffectedWeight;
					info->imR[idx] *= *info->unaffectedWeight;
				}
			}
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
void task_SepStart(separationSubthread *shared_info, size_t task)
{
	separationSubthread *info = &(shared_info[task]);
	// ensure worker is waiting
	pthread_mutex_lock(&(info->work_mtx));
	// set job information & state
	info->state = WORKING;
	// wake-up signal
	pthread_cond_signal(&(info->work_cond));
	pthread_mutex_unlock(&(info->work_mtx));
}
void task_SepWait(separationSubthread *shared_info, size_t task)
{
	separationSubthread *info = &(shared_info[task]);
	while (1)
	{
		pthread_cond_wait(&(info->boss_cond), &(info->boss_mtx));
		if (IDLE == info->state)
			break;
	}
}
void thread_initSep(separationSubthread *shared_info, pthread_t *threads, int nCore)
{
	int i;
	for (i = 0; i < nCore; i++)
	{
		separationSubthread *info = &(shared_info[i + 1]);
		info->state = SETUP;
		pthread_cond_init(&(info->work_cond), NULL);
		pthread_mutex_init(&(info->work_mtx), NULL);
		pthread_cond_init(&(info->boss_cond), NULL);
		pthread_mutex_init(&(info->boss_mtx), NULL);
		pthread_mutex_lock(&(info->boss_mtx));
		pthread_create(&threads[i], NULL, task_Separation, (void *)info);
		task_SepWait(shared_info, i + 1);
	}
}
void thread_exitSep(separationSubthread *shared_info, pthread_t *threads, int nCore)
{
	for (int i = 0; i < nCore; i++)
	{
		separationSubthread *info = &(shared_info[i + 1]);
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
void f32Decompress(float* __restrict out, const uint16_t in)
{
	uint32_t t1 = in & 0x7fff;                       // Non-sign bits
	uint32_t t2 = in & 0x8000;                       // Sign bit
	uint32_t t3 = in & 0x7c00;                       // Exponent
	t1 <<= 13;                              // Align mantissa on MSB
	t2 <<= 16;                              // Shift sign bit into position
	t1 += 0x38000000;                       // Adjust bias
	t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero
	t1 |= t2;                               // Re-insert sign bit
	*((uint32_t*)out) = t1;
}
void* lib2stem_loadCoefficients(void *dat)
{
	void *coeffProvPtr = malloc(getCoeffSize() << 1);
	uint16_t *ptr1 = (uint16_t*)dat;
	float *ptr2 = (float*)coeffProvPtr;
	for (int i = 0; i < sizeof(spleeterQuantized) / sizeof(uint16_t); i++)
		f32Decompress(&ptr2[i], ptr1[i]);
	return coeffProvPtr;
}
void processMT(size_t framesThreading, size_t analyseBinLimit, size_t timeStep, size_t spectralframeCount, void *coeffProvPtr, float unaffectedWeight, float *reL, float *imL, float *reR, float *imR, int spleeterMode)
{
	int i;
	if (framesThreading == 1)
	{
		// Single thread mode
		spleeter nn = (spleeter)allocateSpleeterStr();
		initSpleeter(nn, analyseBinLimit, timeStep, spleeterMode, coeffProvPtr);
		float *maskPtr = 0;
		getMaskPtr(nn, &maskPtr);
		float *magnitudeSpectrogram = (float*)malloc(timeStep * analyseBinLimit * 2 * sizeof(float));
		size_t flrCnt = spectralframeCount / timeStep;
		size_t lastStart = flrCnt * timeStep;
		size_t lastEnd = spectralframeCount - lastStart;
		size_t nnMaskCursor;
		for (size_t j = 0; j < flrCnt; j++)
		{
			size_t nnStep = j * timeStep * FFTSIZE;
			for (nnMaskCursor = 0; nnMaskCursor < timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					magnitudeSpectrogram[0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reL[idx], imL[idx]) * (float)FFTSIZE;
					magnitudeSpectrogram[1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reR[idx], imR[idx]) * (float)FFTSIZE;
				}
			}
			processSpleeter(nn, magnitudeSpectrogram, maskPtr);
			for (nnMaskCursor = 0; nnMaskCursor < timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					float maskL = maskPtr[0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
					float maskR = maskPtr[1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
					reL[idx] *= maskL;
					imL[idx] *= maskL;
					reR[idx] *= maskR;
					imR[idx] *= maskR;
				}
				for (; i < HALFWNDLEN; i++)
				{
					size_t idx = offset + i;
					reL[idx] *= unaffectedWeight;
					imL[idx] *= unaffectedWeight;
					reR[idx] *= unaffectedWeight;
					imR[idx] *= unaffectedWeight;
				}
			}
		}
		size_t nnStep = flrCnt * timeStep * FFTSIZE;
		for (nnMaskCursor = 0; nnMaskCursor < lastEnd; nnMaskCursor++)
		{
			size_t offset = nnStep + nnMaskCursor * FFTSIZE;
			for (i = 0; i < analyseBinLimit; i++)
			{
				size_t idx = offset + i;
				magnitudeSpectrogram[0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reL[idx], imL[idx]) * (float)FFTSIZE;
				magnitudeSpectrogram[1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reR[idx], imR[idx]) * (float)FFTSIZE;
			}
		}
		for (nnMaskCursor = lastEnd; nnMaskCursor < timeStep; nnMaskCursor++)
		{
			for (i = 0; i < analyseBinLimit; i++)
			{
				magnitudeSpectrogram[0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = 0.0f;
				magnitudeSpectrogram[1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = 0.0f;
			}
		}
		processSpleeter(nn, magnitudeSpectrogram, maskPtr);
		for (nnMaskCursor = 0; nnMaskCursor < lastEnd; nnMaskCursor++)
		{
			size_t offset = nnStep + nnMaskCursor * FFTSIZE;
			for (i = 0; i < analyseBinLimit; i++)
			{
				size_t idx = offset + i;
				float maskL = maskPtr[0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
				float maskR = maskPtr[1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
				reL[idx] *= maskL;
				imL[idx] *= maskL;
				reR[idx] *= maskR;
				imR[idx] *= maskR;
			}
			for (; i < HALFWNDLEN; i++)
			{
				size_t idx = offset + i;
				reL[idx] *= unaffectedWeight;
				imL[idx] *= unaffectedWeight;
				reR[idx] *= unaffectedWeight;
				imR[idx] *= unaffectedWeight;
			}
		}
		freeSpleeter(nn);
		free(nn);
		free(magnitudeSpectrogram);
	}
	else
	{
		// Multi thread mode
		size_t flrCnt = spectralframeCount / timeStep;
		size_t lastStart = flrCnt * timeStep;
		size_t lastEnd = spectralframeCount - lastStart;
		separationSubthread *threaddat = (separationSubthread*)malloc(framesThreading * sizeof(separationSubthread));
		pthread_t *th = (pthread_t*)malloc((framesThreading - 1) * sizeof(pthread_t));
		size_t taskPerThread = flrCnt / framesThreading;
		spleeter *nn = (spleeter*)malloc(framesThreading * sizeof(spleeter*));
		float **maskPtr = (float**)malloc(framesThreading * sizeof(float*));
		float **magnitudeSpectrogram = (float**)malloc(framesThreading * sizeof(float*));
		for (i = 0; i < framesThreading; i++)
		{
			nn[i] = (spleeter)allocateSpleeterStr();
			initSpleeter(nn[i], analyseBinLimit, timeStep, spleeterMode, coeffProvPtr);
			getMaskPtr(nn[i], &maskPtr[i]);
			magnitudeSpectrogram[i] = (float*)malloc(timeStep * analyseBinLimit * 2 * sizeof(float));
			threaddat[i].rangeMin = i * taskPerThread;
			if (i < framesThreading - 1)
				threaddat[i].rangeMax = i * taskPerThread + taskPerThread;
			threaddat[i].analyseBinLimit = &analyseBinLimit;
			threaddat[i].timeStep = &timeStep;
			threaddat[i].nn = nn[i];
			threaddat[i].unaffectedWeight = &unaffectedWeight;
			threaddat[i].reL = reL;
			threaddat[i].imL = imL;
			threaddat[i].reR = reR;
			threaddat[i].imR = imR;
			threaddat[i].magnitudeSpectrogram = magnitudeSpectrogram[i];
			threaddat[i].maskPtr = maskPtr[i];
		}
		threaddat[framesThreading - 1].rangeMax = flrCnt;
		thread_initSep(threaddat, th, framesThreading - 1);
		for (i = 0; i < framesThreading - 1; i++)
			task_SepStart(threaddat, i + 1);
		size_t nnMaskCursor;
		for (size_t j = threaddat[0].rangeMin; j < threaddat[0].rangeMax; j++)
		{
			size_t nnStep = j * timeStep * FFTSIZE;
			for (nnMaskCursor = 0; nnMaskCursor < timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					magnitudeSpectrogram[0][0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reL[idx], imL[idx]) * (float)FFTSIZE;
					magnitudeSpectrogram[0][1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reR[idx], imR[idx]) * (float)FFTSIZE;
				}
			}
			processSpleeter(nn[0], magnitudeSpectrogram[0], maskPtr[0]);
			for (nnMaskCursor = 0; nnMaskCursor < timeStep; nnMaskCursor++)
			{
				size_t offset = nnStep + nnMaskCursor * FFTSIZE;
				for (i = 0; i < analyseBinLimit; i++)
				{
					size_t idx = offset + i;
					float maskL = maskPtr[0][0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
					float maskR = maskPtr[0][1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
					reL[idx] *= maskL;
					imL[idx] *= maskL;
					reR[idx] *= maskR;
					imR[idx] *= maskR;
				}
				for (; i < HALFWNDLEN; i++)
				{
					size_t idx = offset + i;
					reL[idx] *= unaffectedWeight;
					imL[idx] *= unaffectedWeight;
					reR[idx] *= unaffectedWeight;
					imR[idx] *= unaffectedWeight;
				}
			}
		}

		size_t nnStep = flrCnt * timeStep * FFTSIZE;
		for (nnMaskCursor = 0; nnMaskCursor < lastEnd; nnMaskCursor++)
		{
			size_t offset = nnStep + nnMaskCursor * FFTSIZE;
			for (i = 0; i < analyseBinLimit; i++)
			{
				size_t idx = offset + i;
				magnitudeSpectrogram[0][0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reL[idx], imL[idx]) * (float)FFTSIZE;
				magnitudeSpectrogram[0][1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = hypotf(reR[idx], imR[idx]) * (float)FFTSIZE;
			}
		}
		for (nnMaskCursor = lastEnd; nnMaskCursor < timeStep; nnMaskCursor++)
		{
			for (i = 0; i < analyseBinLimit; i++)
			{
				magnitudeSpectrogram[0][0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = 0.0f;
				magnitudeSpectrogram[0][1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i] = 0.0f;
			}
		}
		processSpleeter(nn[0], magnitudeSpectrogram[0], maskPtr[0]);
		for (nnMaskCursor = 0; nnMaskCursor < lastEnd; nnMaskCursor++)
		{
			size_t offset = nnStep + nnMaskCursor * FFTSIZE;
			for (i = 0; i < analyseBinLimit; i++)
			{
				size_t idx = offset + i;
				float maskL = maskPtr[0][0 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
				float maskR = maskPtr[0][1 * (analyseBinLimit * timeStep) + analyseBinLimit * nnMaskCursor + i];
				reL[idx] *= maskL;
				imL[idx] *= maskL;
				reR[idx] *= maskR;
				imR[idx] *= maskR;
			}
			for (; i < HALFWNDLEN; i++)
			{
				size_t idx = offset + i;
				reL[idx] *= unaffectedWeight;
				imL[idx] *= unaffectedWeight;
				reR[idx] *= unaffectedWeight;
				imR[idx] *= unaffectedWeight;
			}
		}
		for (i = 0; i < framesThreading - 1; i++)
			task_SepWait(threaddat, i + 1);
		thread_exitSep(threaddat, th, framesThreading - 1);
		for (i = 0; i < framesThreading; i++)
		{
			freeSpleeter(nn[i]);
			free(nn[i]);
			free(magnitudeSpectrogram[i]);
		}
		free(threaddat);
		free(th);
		free(nn);
		free(magnitudeSpectrogram);
		free(maskPtr);
	}
}
extern void openblas_set_num_threads(int num_threads);
#ifndef MAX_PATH
#define MAX_PATH 512
#endif
//#define PLAINFILELOADING
#include "model.c"
int main(int argc, char *argv[])
{
#if _WIN32 == 1 || _WIN64 == 1
	char *backendInfo = "Using Intel MKL 2020 GEMM backend";
#elif __APPLE__ == 1
	char *backendInfo = "Using Accelerate GEMM backend";
#else
	char *backendInfo = "Using OpenBLAS GEMM backend";
	openblas_set_num_threads(1);
#endif
	printf("Computation pipeline written by James Fung\n%s\nFLAC, WAV, MP3 loading supported\n", backendInfo);
	double startTimer = get_wall_time();
	decompressedCoefficients = (float*)malloc(22438 * sizeof(float));
	decompressResamplerMQ(compressedCoeffMQ, decompressedCoefficients);
	int i;
	unsigned int channels;
	drwav_uint64 totalPCMFrameCount;
	char fileIn[2048] = "sgscSmall.wav";
	int twoOrThreeOutput = 1;
	size_t framesThreading = 3;
	size_t analyseBinLimit = 1024;
	size_t timeStep = 512;
#ifndef PLAINFILELOADING
	if (argc < 5)
	{
		free(decompressedCoefficients);
		printf("Invalid program arguments.\nExample:\n%s spawnNthreads timeStep analyseBinLimit stems audioFile\n%s 3 512 1024 3 musicFile.flac\n", argv[0], argv[0]);
		return -2;
	}
	int val0 = atoi(argv[1]);
	if (val0 < 1)
	{
		val0 = 1;
		printf("spawnNthreads clamp to 1\n");
	}
	else
		framesThreading = (size_t)val0;
	if (framesThreading > 4u)
		printf("May causing system go out-of-memory, continue!\n");
	int val1 = atoi(argv[2]);
	int val2 = atoi(argv[3]);
	int val3 = atoi(argv[4]);
	if (val3 <= 2)
		twoOrThreeOutput = 0;
	strncpy(fileIn, argv[5], 2048);
	if (val1 < 64)
	{
		timeStep = 64u;
		printf("timeStep clamp to 64\n");
	}
	else
		timeStep = (size_t)val1;
	if (val2 < 512)
	{
		analyseBinLimit = 512u;
		printf("analyseBinLimit clamp to 512\n");
	}
	else
		analyseBinLimit = (size_t)val2;
	if (!ispowerof2(timeStep))
		printf("Value should be power of 2 or you know what you are typing\nAccepting value and continue\n");
	if (!ispowerof2(analyseBinLimit))
		printf("Value should be power of 2 or you know what you are typing\nAccepting value and continue\n");
	if (analyseBinLimit > 2048)
	{
		analyseBinLimit = 2048;
		printf("Analysis bin limit reached, clamp value to 2048\n");
	}
	float *pSampleData = loadAudioFile(fileIn, 44100.0, &channels, &totalPCMFrameCount);
#else
	float *pSampleData = loadAudioFile(fileIn, 44100.0, &channels, &totalPCMFrameCount);
#endif
	if (!pSampleData)
	{
		free(decompressedCoefficients);
		return -1;
	}
	free(decompressedCoefficients);
	void *coeffProvPtr2 = lib2stem_loadCoefficients((void*)coeffQuantized);
	void *coeffProvPtr1 = (void*)(((char*)coeffProvPtr2) + sizeof(spleeterCoeff));
	char *filename = basenameM(fileIn);
	int readcount = (unsigned int)ceil((float)totalPCMFrameCount / (float)FFTSIZE);
	int finalSize = FFTSIZE * readcount + (FFTSIZE << 1);
	float *splittedBuffer[2];
	splittedBuffer[0] = (float*)calloc(finalSize, sizeof(float));
	splittedBuffer[1] = (float*)calloc(finalSize, sizeof(float));
	channel_splitFloat(pSampleData, (unsigned int)totalPCMFrameCount, splittedBuffer, channels, FFTSIZE);
	if (channels == 1)
		memcpy(splittedBuffer[1], splittedBuffer[0], finalSize * sizeof(float));
	free(pSampleData);
	// Spleeter
	printf("Audio & model file and sample rate conversion takes: %1.14lf sec\n", get_wall_time() - startTimer);
	float unaffectedWeight = 0.1f;
	// STFT forward
	OfflineSTFT *st = (OfflineSTFT*)malloc(sizeof(OfflineSTFT));
	InitSTFT(st, framesThreading);
	float *reL = 0, *imL = 0, *reR = 0, *imR = 0;
	size_t spectralframeCount = stft(st, splittedBuffer[0], splittedBuffer[1], finalSize, &reL, &imL, &reR, &imR);
	if (!twoOrThreeOutput)
	{
		startTimer = get_wall_time();
		processMT(framesThreading, analyseBinLimit, timeStep, spectralframeCount, coeffProvPtr1, unaffectedWeight, reL, imL, reR, imR, 0);
		printf("Inference neural networks using %d cores takes %1.14lf sec\n", (int)framesThreading, get_wall_time() - startTimer);
		float *out1L = 0, *out1R = 0;
		size_t outLen = istft(st, reL, imL, reR, imR, spectralframeCount, &out1L, &out1R);
		free(reL);
		free(imL);
		free(reR);
		free(imR);
		FreeSTFT(st);
		free(st);
		float *out2L = (float*)malloc(outLen * sizeof(float));
		float *out2R = (float*)malloc(outLen * sizeof(float));
		for (i = 0; i < finalSize; i++)
		{
			out2L[i] = splittedBuffer[0][i] - out1L[i];
			out2R[i] = splittedBuffer[1][i] - out1R[i];
		}
		startTimer = get_wall_time();
		float *outBuffer[4] = { out1L, out1R, out2L, out2R };
		//	system("pause");
		free(splittedBuffer[0]);
		free(splittedBuffer[1]);
		size_t totalFrames = totalPCMFrameCount * 2u;
		float *sndBuf1 = (float*)calloc(totalFrames, sizeof(float));
		channel_joinFloat(outBuffer, 2, sndBuf1, totalPCMFrameCount, FFTSIZE);
		float *sndBuf2 = (float*)calloc(totalFrames, sizeof(float));
		channel_joinFloat(&outBuffer[2], 2, sndBuf2, totalPCMFrameCount, FFTSIZE);
		for (i = 0; i < 4; i++)
			free(outBuffer[i]);
		// Vocal
		size_t bufsz = snprintf(NULL, 0, "%s_Vocal.wav", filename);
		char *filenameNew = (char*)malloc(bufsz + 1);
		snprintf(filenameNew, bufsz + 1, "%s_Vocal.wav", filename);
		drwav pWav;
		drwav_data_format format;
		format.container = drwav_container_riff;
		format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
		format.channels = 2;
		format.sampleRate = 44100;
		format.bitsPerSample = 32;
		unsigned int fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
		drwav_uint64 framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf1);
		drwav_uninit(&pWav);
		printf("Saving file -> %s takes %1.14lf sec\n", filenameNew, get_wall_time() - startTimer);
		free(sndBuf1);
		free(filenameNew);
		// Acc
		bufsz = snprintf(NULL, 0, "%s_Accompaniment.wav", filename);
		filenameNew = (char*)malloc(bufsz + 1);
		snprintf(filenameNew, bufsz + 1, "%s_Accompaniment.wav", filename);
		free(filename);
		format.container = drwav_container_riff;
		format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
		format.channels = 2;
		format.sampleRate = 44100;
		format.bitsPerSample = 32;
		fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
		framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf2);
		drwav_uninit(&pWav);
		printf("Saving file -> %s takes %1.14lf sec\n", filenameNew, get_wall_time() - startTimer);
		free(filenameNew);
		free(sndBuf2);
	}
	else
	{
		free(splittedBuffer[0]);
		free(splittedBuffer[1]);
		float *orig_reL = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *orig_imL = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *orig_reR = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *orig_imR = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(orig_reL, reL, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(orig_imL, imL, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(orig_reR, reR, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(orig_imR, imR, spectralframeCount * FFTSIZE * sizeof(float));
		startTimer = get_wall_time();
		processMT(framesThreading, analyseBinLimit, timeStep, spectralframeCount, coeffProvPtr2, unaffectedWeight, reL, imL, reR, imR, 1);
		printf("Inference neural networks using %d cores takes %1.14lf sec\n", (int)framesThreading, get_wall_time() - startTimer);
		for (i = 0; i < spectralframeCount * FFTSIZE; i++)
		{
			orig_reL[i] = orig_reL[i] - reL[i];
			orig_imL[i] = orig_imL[i] - imL[i];
			orig_reR[i] = orig_reR[i] - reR[i];
			orig_imR[i] = orig_imR[i] - imR[i];
		}
		float *out1L = 0, *out1R = 0;
		size_t outLen = istft(st, reL, imL, reR, imR, spectralframeCount, &out1L, &out1R);
		free(reL);
		free(imL);
		free(reR);
		free(imR);
		float *out_Acc_VocalL = 0, *out_Acc_VocalR = 0;
		float *accvocal_reL = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *accvocal_imL = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *accvocal_reR = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		float *accvocal_imR = (float*)malloc(spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(accvocal_reL, orig_reL, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(accvocal_imL, orig_imL, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(accvocal_reR, orig_reR, spectralframeCount * FFTSIZE * sizeof(float));
		memcpy(accvocal_imR, orig_imR, spectralframeCount * FFTSIZE * sizeof(float));
		outLen = istft(st, orig_reL, orig_imL, orig_reR, orig_imR, spectralframeCount, &out_Acc_VocalL, &out_Acc_VocalR);
		free(orig_reL);
		free(orig_reR);
		free(orig_imL);
		free(orig_imR);
		size_t totalFrames = totalPCMFrameCount * 2u;
		float *sndBuf = (float*)calloc(totalFrames, sizeof(float));
		float *ob1[2] = { out1L, out1R };
		channel_joinFloat(ob1, 2, sndBuf, totalPCMFrameCount, FFTSIZE);
		free(out1L);
		free(out1R);
		// Drum
		size_t bufsz = snprintf(NULL, 0, "%s_Drum.wav", filename);
		char *filenameNew = (char*)malloc(bufsz + 1);
		snprintf(filenameNew, bufsz + 1, "%s_Drum.wav", filename);
		drwav pWav;
		drwav_data_format format;
		format.container = drwav_container_riff;
		format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
		format.channels = 2;
		format.sampleRate = 44100;
		format.bitsPerSample = 32;
		unsigned int fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
		drwav_uint64 framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf);
		drwav_uninit(&pWav);
		printf("Saving file -> %s takes %1.14lf sec\n", filenameNew, get_wall_time() - startTimer);
		free(filenameNew);

		startTimer = get_wall_time();
		processMT(framesThreading, analyseBinLimit, timeStep, spectralframeCount, coeffProvPtr1, unaffectedWeight, accvocal_reL, accvocal_imL, accvocal_reR, accvocal_imR, 0);
		printf("Inference neural networks using %d cores takes %1.14lf sec\n", (int)framesThreading, get_wall_time() - startTimer);
		float *out2L = 0, *out2R = 0;
		outLen = istft(st, accvocal_reL, accvocal_imL, accvocal_reR, accvocal_imR, spectralframeCount, &out2L, &out2R);
		free(accvocal_reL);
		free(accvocal_imL);
		free(accvocal_reR);
		free(accvocal_imR);

		FreeSTFT(st);
		free(st);
		float *out3L = (float*)malloc(outLen * sizeof(float));
		float *out3R = (float*)malloc(outLen * sizeof(float));
		for (i = 0; i < finalSize; i++)
		{
			out3L[i] = out_Acc_VocalL[i] - out2L[i];
			out3R[i] = out_Acc_VocalR[i] - out2R[i];
		}
		free(out_Acc_VocalL);
		free(out_Acc_VocalR);
		ob1[0] = out3L;
		ob1[1] = out3R;
		channel_joinFloat(ob1, 2, sndBuf, totalPCMFrameCount, FFTSIZE);
		free(out3L);
		free(out3R);
		bufsz = snprintf(NULL, 0, "%s_Accompaniment.wav", filename);
		filenameNew = (char*)malloc(bufsz + 1);
		snprintf(filenameNew, bufsz + 1, "%s_Accompaniment.wav", filename);
		format.container = drwav_container_riff;
		format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
		format.channels = 2;
		format.sampleRate = 44100;
		format.bitsPerSample = 32;
		fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
		framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf);
		drwav_uninit(&pWav);
		printf("Saving file -> %s takes %1.14lf sec\n", filenameNew, get_wall_time() - startTimer);
		free(filenameNew);

		ob1[0] = out2L;
		ob1[1] = out2R;
		channel_joinFloat(ob1, 2, sndBuf, totalPCMFrameCount, FFTSIZE);
		bufsz = snprintf(NULL, 0, "%s_Vocal.wav", filename);
		filenameNew = (char*)malloc(bufsz + 1);
		snprintf(filenameNew, bufsz + 1, "%s_Vocal.wav", filename);
		free(filename);
		format.container = drwav_container_riff;
		format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
		format.channels = 2;
		format.sampleRate = 44100;
		format.bitsPerSample = 32;
		fail = drwav_init_file_write(&pWav, filenameNew, &format, 0);
		framesWritten = drwav_write_pcm_frames(&pWav, totalPCMFrameCount, sndBuf);
		drwav_uninit(&pWav);
		printf("Saving file -> %s takes %1.14lf sec\n", filenameNew, get_wall_time() - startTimer);
		free(filenameNew);
		free(out2L);
		free(out2R);
		free(sndBuf);
	}
	free(coeffProvPtr2);
	return 0;
}