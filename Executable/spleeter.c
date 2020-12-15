#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "spleeter.h"
typedef struct
{
	int inputCh, outputCh, height, width, kernelSize, stride, pad, dilation, out_h, out_w, hoffset, woffset;
	float *kernel;
} DLConv;
typedef struct
{
	int inputCh, outputCh, height, width, stride, pad, out_h, out_w, kernelSize, hoffset, woffset;
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
static const float sigmoidTbl[TBL_SIZE + 1] = { 0.00091105f, 0.00092358f, 0.00093628f, 0.00094916f, 0.00096221f, 0.00097545f, 0.00098886f, 0.00100246f, 0.00101624f, 0.00103022f, 0.00104439f, 0.00105875f, 0.00107331f, 0.00108807f, 0.00110303f, 0.00111819f, 0.00113357f, 0.00114916f, 0.00116496f, 0.00118097f, 0.00119721f, 0.00121367f, 0.00123036f, 0.00124727f, 0.00126442f, 0.00128181f, 0.00129943f, 0.00131729f, 0.00133540f, 0.00135376f, 0.00137237f, 0.00139123f, 0.00141036f, 0.00142975f, 0.00144940f, 0.00146932f, 0.00148952f, 0.00150999f, 0.00153075f, 0.00155178f, 0.00157311f, 0.00159473f, 0.00161665f, 0.00163887f, 0.00166139f, 0.00168422f, 0.00170737f, 0.00173083f, 0.00175461f, 0.00177872f, 0.00180317f, 0.00182794f, 0.00185306f, 0.00187852f, 0.00190433f, 0.00193049f, 0.00195702f, 0.00198390f, 0.00201116f, 0.00203879f, 0.00206679f, 0.00209519f, 0.00212397f, 0.00215314f, 0.00218272f, 0.00221270f, 0.00224309f, 0.00227390f, 0.00230513f, 0.00233678f, 0.00236887f, 0.00240141f, 0.00243438f, 0.00246781f, 0.00250170f, 0.00253605f, 0.00257087f, 0.00260617f, 0.00264195f, 0.00267822f, 0.00271499f, 0.00275226f, 0.00279004f, 0.00282834f, 0.00286716f, 0.00290651f, 0.00294640f, 0.00298684f, 0.00302784f, 0.00306939f, 0.00311151f, 0.00315421f, 0.00319749f, 0.00324136f, 0.00328583f, 0.00333091f, 0.00337661f, 0.00342294f, 0.00346989f, 0.00351749f, 0.00356574f, 0.00361464f, 0.00366422f, 0.00371447f, 0.00376541f, 0.00381705f, 0.00386939f, 0.00392245f, 0.00397623f, 0.00403074f, 0.00408600f, 0.00414202f, 0.00419880f, 0.00425635f, 0.00431469f, 0.00437382f, 0.00443377f, 0.00449453f, 0.00455611f, 0.00461854f, 0.00468182f, 0.00474597f, 0.00481098f, 0.00487689f, 0.00494369f, 0.00501140f, 0.00508004f, 0.00514961f, 0.00522013f, 0.00529160f, 0.00536406f, 0.00543750f, 0.00551193f, 0.00558739f, 0.00566386f, 0.00574139f, 0.00581996f, 0.00589960f, 0.00598033f, 0.00606216f, 0.00614509f, 0.00622916f, 0.00631437f, 0.00640073f, 0.00648827f, 0.00657700f, 0.00666693f, 0.00675809f, 0.00685048f, 0.00694413f, 0.00703905f, 0.00713525f, 0.00723276f, 0.00733160f, 0.00743177f, 0.00753331f, 0.00763622f, 0.00774052f, 0.00784624f, 0.00795339f, 0.00806199f, 0.00817206f, 0.00828363f, 0.00839670f, 0.00851130f, 0.00862746f, 0.00874518f, 0.00886450f, 0.00898543f, 0.00910799f, 0.00923221f, 0.00935811f, 0.00948571f, 0.00961504f, 0.00974610f, 0.00987894f, 0.01001357f, 0.01015002f, 0.01028830f, 0.01042845f, 0.01057049f, 0.01071444f, 0.01086033f, 0.01100819f, 0.01115803f, 0.01130989f, 0.01146380f, 0.01161977f, 0.01177784f, 0.01193804f, 0.01210039f, 0.01226491f, 0.01243165f, 0.01260063f, 0.01277187f, 0.01294541f, 0.01312127f, 0.01329949f, 0.01348010f, 0.01366313f, 0.01384861f, 0.01403657f, 0.01422705f, 0.01442007f, 0.01461567f, 0.01481389f, 0.01501475f, 0.01521829f, 0.01542455f, 0.01563356f, 0.01584536f, 0.01605998f, 0.01627746f, 0.01649784f, 0.01672114f, 0.01694742f, 0.01717671f, 0.01740904f, 0.01764446f, 0.01788301f, 0.01812472f, 0.01836964f, 0.01861780f, 0.01886925f, 0.01912403f, 0.01938219f, 0.01964376f, 0.01990879f, 0.02017732f, 0.02044939f, 0.02072506f, 0.02100436f, 0.02128735f, 0.02157407f, 0.02186456f, 0.02215887f, 0.02245705f, 0.02275916f, 0.02306523f, 0.02337531f, 0.02368947f, 0.02400775f, 0.02433019f, 0.02465685f, 0.02498779f, 0.02532306f, 0.02566271f, 0.02600678f, 0.02635535f, 0.02670847f, 0.02706617f, 0.02742854f, 0.02779562f, 0.02816748f, 0.02854415f, 0.02892572f, 0.02931223f, 0.02970375f, 0.03010034f, 0.03050205f, 0.03090896f, 0.03132112f, 0.03173859f, 0.03216145f, 0.03258974f, 0.03302355f, 0.03346293f, 0.03390796f, 0.03435869f, 0.03481520f, 0.03527755f, 0.03574581f, 0.03622006f, 0.03670035f, 0.03718678f, 0.03767939f, 0.03817828f, 0.03868350f, 0.03919514f, 0.03971326f, 0.04023794f, 0.04076926f, 0.04130730f, 0.04185214f, 0.04240383f, 0.04296247f, 0.04352814f, 0.04410091f, 0.04468087f, 0.04526810f, 0.04586267f, 0.04646466f, 0.04707418f, 0.04769128f, 0.04831607f, 0.04894862f, 0.04958903f, 0.05023737f, 0.05089372f, 0.05155819f, 0.05223086f, 0.05291181f, 0.05360114f, 0.05429894f, 0.05500529f, 0.05572029f, 0.05644403f, 0.05717659f, 0.05791809f, 0.05866860f, 0.05942822f, 0.06019705f, 0.06097518f, 0.06176271f, 0.06255973f, 0.06336634f, 0.06418265f, 0.06500873f, 0.06584470f, 0.06669066f, 0.06754670f, 0.06841291f, 0.06928942f, 0.07017630f, 0.07107367f, 0.07198163f, 0.07290028f, 0.07382971f, 0.07477005f, 0.07572138f, 0.07668380f, 0.07765744f, 0.07864238f, 0.07963873f, 0.08064662f, 0.08166611f, 0.08269734f, 0.08374041f, 0.08479541f, 0.08586246f, 0.08694165f, 0.08803311f, 0.08913694f, 0.09025324f, 0.09138211f, 0.09252366f, 0.09367800f, 0.09484524f, 0.09602549f, 0.09721884f, 0.09842541f, 0.09964530f, 0.10087862f, 0.10212547f, 0.10338596f, 0.10466021f, 0.10594828f, 0.10725033f, 0.10856642f, 0.10989668f, 0.11124121f, 0.11260010f, 0.11397346f, 0.11536140f, 0.11676401f, 0.11818139f, 0.11961366f, 0.12106091f, 0.12252321f, 0.12400070f, 0.12549345f, 0.12700157f, 0.12852514f, 0.13006428f, 0.13161905f, 0.13318956f, 0.13477592f, 0.13637818f, 0.13799648f, 0.13963085f, 0.14128143f, 0.14294825f, 0.14463145f, 0.14633107f, 0.14804719f, 0.14977993f, 0.15152934f, 0.15329549f, 0.15507847f, 0.15687835f, 0.15869519f, 0.16052906f, 0.16238004f, 0.16424817f, 0.16613355f, 0.16803622f, 0.16995622f, 0.17189366f, 0.17384852f, 0.17582092f, 0.17781086f, 0.17981842f, 0.18184364f, 0.18388654f, 0.18594719f, 0.18802561f, 0.19012183f, 0.19223589f, 0.19436781f, 0.19651762f, 0.19868536f, 0.20087102f, 0.20307463f, 0.20529620f, 0.20753576f, 0.20979328f, 0.21206881f, 0.21436231f, 0.21667379f, 0.21900325f, 0.22135070f, 0.22371607f, 0.22609939f, 0.22850062f, 0.23091975f, 0.23335676f, 0.23581159f, 0.23828420f, 0.24077460f, 0.24328271f, 0.24580847f, 0.24835183f, 0.25091279f, 0.25349125f, 0.25608709f, 0.25870037f, 0.26133093f, 0.26397869f, 0.26664361f, 0.26932561f, 0.27202454f, 0.27474037f, 0.27747297f, 0.28022227f, 0.28298810f, 0.28577045f, 0.28856909f, 0.29138398f, 0.29421496f, 0.29706192f, 0.29992473f, 0.30280325f, 0.30569732f, 0.30860683f, 0.31153160f, 0.31447145f, 0.31742626f, 0.32039589f, 0.32338011f, 0.32637879f, 0.32939172f, 0.33241874f, 0.33545968f, 0.33851433f, 0.34158251f, 0.34466401f, 0.34775859f, 0.35086608f, 0.35398629f, 0.35711899f, 0.36026391f, 0.36342093f, 0.36658975f, 0.36977011f, 0.37296185f, 0.37616467f, 0.37937838f, 0.38260269f, 0.38583741f, 0.38908216f, 0.39233685f, 0.39560106f, 0.39887467f, 0.40215728f, 0.40544870f, 0.40874869f, 0.41205689f, 0.41537306f, 0.41869688f, 0.42202812f, 0.42536652f, 0.42871168f, 0.43206340f, 0.43542132f, 0.43878520f, 0.44215471f, 0.44552955f, 0.44890940f, 0.45229396f, 0.45568299f, 0.45907614f, 0.46247303f, 0.46587345f, 0.46927702f, 0.47268346f, 0.47609246f, 0.47950369f, 0.48291680f, 0.48633155f, 0.48974755f, 0.49316448f, 0.49658209f, 0.50000000f, 0.50341791f, 0.50683552f, 0.51025248f, 0.51366848f, 0.51708317f, 0.52049631f, 0.52390748f, 0.52731651f, 0.53072298f, 0.53412652f, 0.53752697f, 0.54092389f, 0.54431701f, 0.54770601f, 0.55109060f, 0.55447048f, 0.55784535f, 0.56121480f, 0.56457865f, 0.56793660f, 0.57128835f, 0.57463354f, 0.57797182f, 0.58130306f, 0.58462697f, 0.58794314f, 0.59125131f, 0.59455127f, 0.59784269f, 0.60112536f, 0.60439891f, 0.60766321f, 0.61091781f, 0.61416262f, 0.61739731f, 0.62062162f, 0.62383533f, 0.62703818f, 0.63022989f, 0.63341027f, 0.63657910f, 0.63973606f, 0.64288098f, 0.64601368f, 0.64913392f, 0.65224141f, 0.65533602f, 0.65841752f, 0.66148567f, 0.66454029f, 0.66758120f, 0.67060828f, 0.67362124f, 0.67661995f, 0.67960411f, 0.68257374f, 0.68552858f, 0.68846846f, 0.69139320f, 0.69430268f, 0.69719678f, 0.70007521f, 0.70293802f, 0.70578504f, 0.70861602f, 0.71143085f, 0.71422958f, 0.71701187f, 0.71977776f, 0.72252703f, 0.72525960f, 0.72797543f, 0.73067439f, 0.73335636f, 0.73602128f, 0.73866904f, 0.74129963f, 0.74391288f, 0.74650878f, 0.74908721f, 0.75164813f, 0.75419158f, 0.75671732f, 0.75922543f, 0.76171577f, 0.76418841f, 0.76664323f, 0.76908022f, 0.77149934f, 0.77390057f, 0.77628392f, 0.77864933f, 0.78099674f, 0.78332621f, 0.78563768f, 0.78793120f, 0.79020667f, 0.79246426f, 0.79470378f, 0.79692537f, 0.79912901f, 0.80131465f, 0.80348235f, 0.80563217f, 0.80776411f, 0.80987817f, 0.81197441f, 0.81405276f, 0.81611341f, 0.81815636f, 0.82018155f, 0.82218909f, 0.82417905f, 0.82615143f, 0.82810634f, 0.83004373f, 0.83196384f, 0.83386642f, 0.83575183f, 0.83761996f, 0.83947092f, 0.84130478f, 0.84312171f, 0.84492153f, 0.84670448f, 0.84847069f, 0.85022008f, 0.85195273f, 0.85366899f, 0.85536855f, 0.85705173f, 0.85871857f, 0.86036915f, 0.86200351f, 0.86362177f, 0.86522406f, 0.86681038f, 0.86838096f, 0.86993575f, 0.87147486f, 0.87299842f, 0.87450653f, 0.87599933f, 0.87747681f, 0.87893909f, 0.88038635f, 0.88181859f, 0.88323599f, 0.88463867f, 0.88602656f, 0.88739991f, 0.88875884f, 0.89010334f, 0.89143354f, 0.89274967f, 0.89405167f, 0.89533985f, 0.89661402f, 0.89787447f, 0.89912134f, 0.90035468f, 0.90157461f, 0.90278113f, 0.90397453f, 0.90515476f, 0.90632194f, 0.90747637f, 0.90861797f, 0.90974683f, 0.91086304f, 0.91196692f, 0.91305828f, 0.91413754f, 0.91520458f, 0.91625965f, 0.91730267f, 0.91833389f, 0.91935343f, 0.92036128f, 0.92135757f, 0.92234260f, 0.92331618f, 0.92427868f, 0.92523003f, 0.92617035f, 0.92709976f, 0.92801839f, 0.92892635f, 0.92982370f, 0.93071055f, 0.93158704f, 0.93245327f, 0.93330938f, 0.93415529f, 0.93499130f, 0.93581736f, 0.93663365f, 0.93744022f, 0.93823731f, 0.93902481f, 0.93980294f, 0.94057178f, 0.94133139f, 0.94208187f, 0.94282341f, 0.94355595f, 0.94427973f, 0.94499469f, 0.94570112f, 0.94639885f, 0.94708818f, 0.94776917f, 0.94844174f, 0.94910634f, 0.94976270f, 0.95041102f, 0.95105141f, 0.95168388f, 0.95230865f, 0.95292586f, 0.95353532f, 0.95413738f, 0.95473194f, 0.95531917f, 0.95589906f, 0.95647180f, 0.95703751f, 0.95759624f, 0.95814794f, 0.95869267f, 0.95923072f, 0.95976204f, 0.96028674f, 0.96080494f, 0.96131647f, 0.96182173f, 0.96232057f, 0.96281320f, 0.96329963f, 0.96377999f, 0.96425414f, 0.96472245f, 0.96518475f, 0.96564132f, 0.96609205f, 0.96653706f, 0.96697646f, 0.96741027f, 0.96783853f, 0.96826136f, 0.96867889f, 0.96909106f, 0.96949792f, 0.96989965f, 0.97029626f, 0.97068775f, 0.97107434f, 0.97145587f, 0.97183257f, 0.97220439f, 0.97257149f, 0.97293383f, 0.97329152f, 0.97364467f, 0.97399318f, 0.97433734f, 0.97467697f, 0.97501218f, 0.97534311f, 0.97566980f, 0.97599232f, 0.97631049f, 0.97662467f, 0.97693479f, 0.97724086f, 0.97754294f, 0.97784120f, 0.97813547f, 0.97842592f, 0.97871268f, 0.97899562f, 0.97927493f, 0.97955060f, 0.97982270f, 0.98009127f, 0.98035622f, 0.98061782f, 0.98087597f, 0.98113072f, 0.98138225f, 0.98163038f, 0.98187524f, 0.98211700f, 0.98235554f, 0.98259097f, 0.98282325f, 0.98305261f, 0.98327893f, 0.98350221f, 0.98372251f, 0.98394001f, 0.98415458f, 0.98436642f, 0.98457539f, 0.98478174f, 0.98498523f, 0.98518616f, 0.98538429f, 0.98557997f, 0.98577291f, 0.98596340f, 0.98615140f, 0.98633689f, 0.98651993f, 0.98670048f, 0.98687869f, 0.98705459f, 0.98722816f, 0.98739934f, 0.98756832f, 0.98773509f, 0.98789960f, 0.98806202f, 0.98822218f, 0.98838019f, 0.98853612f, 0.98869008f, 0.98884189f, 0.98899186f, 0.98913974f, 0.98928553f, 0.98942953f, 0.98957157f, 0.98971164f, 0.98984993f, 0.98998636f, 0.99012113f, 0.99025387f, 0.99038494f, 0.99051428f, 0.99064189f, 0.99076778f, 0.99089199f, 0.99101454f, 0.99113548f, 0.99125481f, 0.99137259f, 0.99148870f, 0.99160331f, 0.99171633f, 0.99182796f, 0.99193794f, 0.99204659f, 0.99215370f, 0.99225944f, 0.99236381f, 0.99246663f, 0.99256825f, 0.99266839f, 0.99276721f, 0.99286473f, 0.99296099f, 0.99305588f, 0.99314958f, 0.99324185f, 0.99333304f, 0.99342304f, 0.99351174f, 0.99359930f, 0.99368566f, 0.99377090f, 0.99385482f, 0.99393785f, 0.99401975f, 0.99410039f, 0.99418008f, 0.99425864f, 0.99433607f, 0.99441260f, 0.99448806f, 0.99456257f, 0.99463588f, 0.99470842f, 0.99477994f, 0.99485034f, 0.99491996f, 0.99498862f, 0.99505627f, 0.99512309f, 0.99518895f, 0.99525404f, 0.99531811f, 0.99538147f, 0.99544394f, 0.99550545f, 0.99556619f, 0.99562621f, 0.99568534f, 0.99574369f, 0.99580115f, 0.99585801f, 0.99591404f, 0.99596930f, 0.99602377f, 0.99607760f, 0.99613059f, 0.99618298f, 0.99623460f, 0.99628556f, 0.99633574f, 0.99638534f, 0.99643421f, 0.99648249f, 0.99653012f, 0.99657708f, 0.99662340f, 0.99666911f, 0.99671423f, 0.99675864f, 0.99680245f, 0.99684578f, 0.99688846f, 0.99693066f, 0.99697220f, 0.99701309f, 0.99705362f, 0.99709344f, 0.99713278f, 0.99717170f, 0.99720997f, 0.99724776f, 0.99728501f, 0.99732178f, 0.99735802f, 0.99739385f, 0.99742907f, 0.99746394f, 0.99749833f, 0.99753213f, 0.99756563f, 0.99759859f, 0.99763107f, 0.99766326f, 0.99769491f, 0.99772614f, 0.99775690f, 0.99778724f, 0.99781728f, 0.99784684f, 0.99787605f, 0.99790478f, 0.99793327f, 0.99796116f, 0.99798882f, 0.99801612f, 0.99804294f, 0.99806958f, 0.99809569f, 0.99812144f, 0.99814701f, 0.99817204f, 0.99819690f, 0.99822122f, 0.99824536f, 0.99826920f, 0.99829262f, 0.99831581f, 0.99833858f, 0.99836117f, 0.99838340f, 0.99840528f, 0.99842691f, 0.99844813f, 0.99846917f, 0.99848998f, 0.99851042f, 0.99853063f, 0.99855059f, 0.99857020f, 0.99858958f, 0.99860877f, 0.99862766f, 0.99864620f, 0.99866462f, 0.99868268f, 0.99870050f, 0.99871826f, 0.99873561f, 0.99875271f, 0.99876958f, 0.99878639f, 0.99880278f, 0.99881905f, 0.99883503f, 0.99885082f, 0.99886644f, 0.99888176f, 0.99889696f, 0.99891198f, 0.99892670f, 0.99894124f, 0.99895561f, 0.99896979f, 0.99898368f, 0.99899751f, 0.99901116f, 0.99902451f, 0.99903786f, 0.99905080f, 0.99906379f, 0.99907637f, 0.99908900f, 1.00000000f };
float fastSigmoid(float x)
{
	if (x > 7.0f)
		return 1.0f;
	else if (x < -7.0f)
		return 0.0f;
	else
	{
		short Index = (short)((x + 7.0f) / 0.01367188f);
		float X1 = -7.0f + 0.01367188f * Index;
		return sigmoidTbl[Index] + (sigmoidTbl[Index + 1] - sigmoidTbl[Index]) / (-7.0f + 0.01367188f * (Index + 1) - X1) * (x - X1);
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
	if (x < -15.0f) // Too small, it's whole of junk, denormal hurts
		return -1.0f;
	return x >= 0.0f ? x : expf(x) - 1.0f;
}
#include "conv_util.h"
void initTransposeConv2dLayer(DLtransposedConv *ly, int inputCh, int outputCh, int height, int width, int stride, int pad, int outputPadding, float *kernel, int kernelSize, int hoffset, int woffset)
{
	ly->inputCh = inputCh;
	ly->outputCh = outputCh;
	ly->height = height;
	ly->width = width;
	ly->stride = stride;
	ly->kernel = kernel;
	ly->kernelSize = kernelSize;
	ly->hoffset = hoffset;
	ly->woffset = woffset;
	ly->pad = pad;
	ly->out_h = transpconv_out_dim(height, kernelSize, stride, pad, outputPadding);
	ly->out_w = transpconv_out_dim(width, kernelSize, stride, pad, outputPadding);
}
void processTransposeConv2dLayer(DLtransposedConv *ly, float *x, float *y, float *workspace)
{
	gemm(1, 0, ly->kernelSize * ly->kernelSize * ly->outputCh, ly->height * ly->width, ly->inputCh, 1.0f, ly->kernel, ly->kernelSize * ly->kernelSize * ly->outputCh, x, ly->height * ly->width, 0.0f, workspace, ly->height * ly->width);
	memset(y, 0, ly->out_w * ly->out_h * ly->outputCh * sizeof(float));
	col2im_dilated_cpu(workspace, ly->outputCh, ly->out_h, ly->out_w, ly->kernelSize, ly->stride, ly->pad, y, ly->hoffset, ly->woffset);
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
void initSpleeter(struct _spleeter *nn, size_t width, size_t height, int stemMode, void *coeff)
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
	initTransposeConv2dLayer(&nn->tpconv[0], 512, 256, nn->convLy[5].out_h, nn->convLy[5].out_w, stride, stride, 1, nn->coeffProvider->up1_transp_convWeight, kernelSize, 1, 1);
	initTransposeConv2dLayer(&nn->tpconv[1], 512, 128, nn->tpconv[0].out_h, nn->tpconv[0].out_w, stride, stride, 1, nn->coeffProvider->up2_transp_convWeight, kernelSize, 1, 1);
	initTransposeConv2dLayer(&nn->tpconv[2], 256, 64, nn->tpconv[1].out_h, nn->tpconv[1].out_w, stride, stride, 1, nn->coeffProvider->up3_transp_convWeight, kernelSize, 1, 1);
	initTransposeConv2dLayer(&nn->tpconv[3], 128, 32, nn->tpconv[2].out_h, nn->tpconv[2].out_w, stride, stride, 1, nn->coeffProvider->up4_transp_convWeight, kernelSize, 1, 1);
	initTransposeConv2dLayer(&nn->tpconv[4], 64, 16, nn->tpconv[3].out_h, nn->tpconv[3].out_w, stride, stride, 1, nn->coeffProvider->up5_transp_convWeight, kernelSize, 1, 1);
	initTransposeConv2dLayer(&nn->tpconv[5], 32, 1, nn->tpconv[4].out_h, nn->tpconv[4].out_w, stride, stride, 1, nn->coeffProvider->up6_transp_convWeight, kernelSize, 1, 1);
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
			y[s * nn->whProd + i] = fastSigmoid(nn->catLayer[s * nn->whProd + i] + nn->coeffProvider->up7_convBias[s]);
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
