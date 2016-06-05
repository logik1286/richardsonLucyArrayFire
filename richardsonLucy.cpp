/*
The MIT License (MIT)

Copyright (c) 2016 logik1286@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <arrayfire.h>
#include <iostream>
#include <string.h>
#include <memory>
#include <vector>
#include <map>
#include <cfloat>

using namespace std;

typedef const char _TCHAR;

string getArgument(int argc, _TCHAR* argv[], int argIndex){

	if (argIndex >= argc){
		//Invalid index returns NULL string
		throw argIndex;
	}

	short stringSize = 0;
	while (*(argv[argIndex] + stringSize) != '\0' || stringSize < 0) stringSize++;

	if (stringSize <= 0) {
		//Terminate character was not found (or string is very long). Abort to be safe.
		throw argIndex;
	}

	return string(argv[argIndex], argv[argIndex] + stringSize);

}

template <typename T> struct MemStruct{

	std::shared_ptr <T> getMem(){ return mem; };
	std::vector<size_t> getDims(){ return dims; };

	size_t getTotalElements(){
		return totalElements;
	};

	size_t getSizeBytes(){
		return totalElements * sizeof(T);
	}

	MemStruct<T>(std::vector<size_t> _dims) : dims(_dims){

		totalElements = 1;

		for (auto & i : dims)
			totalElements *= i;

		mem = std::shared_ptr<T>(new T[totalElements], std::default_delete<T[]>());

	};

	MemStruct<T>() : totalElements(0){};

private:

	size_t totalElements;
	std::shared_ptr<T> mem;
	std::vector<size_t> dims;

};

int loadInputData(string inputPath, size_t dimXY, MemStruct<float> & rtn){

	vector<size_t> dims = { dimXY, dimXY };

	rtn = MemStruct<float>(dims);

	if (rtn.getMem().get() == nullptr)
		return -1;

	FILE * file = nullptr;
	file = fopen(inputPath.c_str(), "rb+");

	if (file == nullptr)
		return -3;

	int read = fread(rtn.getMem().get(), sizeof(float), rtn.getTotalElements(), file);
	fclose(file);

	if (read != rtn.getTotalElements())
		return -2;

	return 0;

}

int saveDataToFile(string inputPath, MemStruct<float> & data){

	FILE * file = nullptr;
	file = fopen(inputPath.c_str(), "wb+");

	if (file == nullptr)
		return -1;

	int wrote = fwrite(data.getMem().get(), sizeof(float), data.getTotalElements(), file);
	fclose(file);

	if (wrote != data.getTotalElements())
		return -2;

	return 0;

}

class RollingFileSave{

public:
	RollingFileSave(){
		fHandle = nullptr;
		path = "";
	}

	RollingFileSave(std::string _path) :path(_path){

		fHandle = nullptr;

		fHandle = fopen(path.c_str(), "wb+");

		if (fHandle == nullptr)
			throw - 1;

	};

	int writeData(MemStruct<float> data){

		if (fHandle == nullptr)
			return -1;

		int wrote = fwrite(data.getMem().get(), sizeof(float), data.getTotalElements(), fHandle);

		if (wrote != data.getTotalElements())
			return -2;

	}

	~RollingFileSave(){
		if (fHandle != nullptr) fclose(fHandle);
	}

private:

	FILE * fHandle;
	std::string path;
};


int generatePSF(size_t size, float sigma, MemStruct<float> & rtn){

	size_t paddedSize = size + (size + 1) % 2; //Need to keep odd size. Otherwise RL blows up.
	//Think it has to do with the center of the PSF not being handled correctly during cross correlation.

	vector<size_t> dims = { paddedSize, paddedSize };

	rtn = MemStruct<float>(dims);

	if (rtn.getMem().get() == nullptr)
		return -1;

	float sum = 0;

	//zero out everything
	memset(rtn.getMem().get(), 0, rtn.getTotalElements()  * sizeof(float));

	//Compute psf
	for (auto y = 0; y < size; y++){

		float yCoord = (y - (size - 1) / 2.0f) / (2.0f * sigma);
		int idx = y * paddedSize;

		for (auto x = 0; x < size; x++){

			float xCoord = (x - (size - 1) / 2.0f) / (2.0f * sigma);
			float val = exp(-(yCoord * yCoord + xCoord * xCoord));
			rtn.getMem().get()[idx + x] = val;
			sum += val;
		}

	}

	//Normalize
	for (auto y = 0; y < size; y++){
		int idx = y * paddedSize;
		for (auto x = 0; x < size; x++){
			rtn.getMem().get()[idx + x] /= sum;
		}

	}

	return 0;

}

//#define BENCHMARK

#ifdef BENCHMARK
#define BENCHMARK_ITER 1

#define RUN_BENCHMARK(accum, rtn, fxn) \
	{\
	af::sync();\
	af::timer time = af::timer::start(); \
	for(auto i = 0; i < BENCHMARK_ITER; i++){\
		rtn = fxn;\
		rtn.eval();\
	};\
	af::sync();\
	accum += af::timer::stop(time) / BENCHMARK_ITER; \
	}

#else

#define RUN_BENCHMARK(accum, rtn, fxn) rtn = fxn;

#endif

int runRL(MemStruct<float> psfImage, MemStruct<float> inputImage, MemStruct<float> & simulatedImage, MemStruct<float> & outputImage, int iterations, string rollingSave = ""){

	RollingFileSave saveEst(rollingSave + ".estimate");

	try{

		auto imageDims = inputImage.getDims();
		auto psfDims = psfImage.getDims();

		//Create arrays to store simulation inputs
		af::array orig(imageDims[0], imageDims[1], (float *)inputImage.getMem().get());
		af::array psf(psfDims[0], psfDims[1], (float *)psfImage.getMem().get());
		orig = 1; orig.eval();
		//Initialize arrays used to return simulation outputs to calling routine
		simulatedImage = MemStruct<float>(imageDims);
		outputImage = MemStruct<float>(imageDims);

		const float minVal = FLT_MIN;
		const float ratioMax = 100;
		const float ratioMin = .001;
		//Clip negative values to zero
		orig = af::max(minVal, orig);

		//"Simulate" camera blur
		af::array measured = af::max(minVal, af::convolve2(orig, psf, af::convMode::AF_CONV_DEFAULT)); //max necessary since af may choose to use fft conv, which can create negative values

		//Initialize algorithm with acquired image
		af::array est(measured);
		af::array estOut(est);

		//Create flipped PSF in lieu of having a cross-correlate function 
		af::array psfUD = af::flip(psf, 0);
		psfUD = af::flip(psfUD, 1);

		//Create intermediate variables
		af::array forward(measured);
		af::array denom(measured);
		af::array ratio(measured); ratio = 1.0f; ratio.eval();
		af::array ratioClampL(ratio);
		af::array ratioClampU(ratio);
		af::array update(ratio);

		map < string, double > timers;

		timers["forward"] = 0;
		timers["clamp1"] = 0;
		timers["ratio"] = 0;
		timers["clamp2"] = 0;
		timers["clamp3"] = 0;
		timers["update"] = 0;
		timers["mply"] = 0;
		timers["reinit"] = 0;
		af::sync();

		af::timer loopTime = af::timer::start();

		//Run iterations
		for (int i = 0; i < iterations; i++){
			cout << "Iteration " << i << endl;
                        //restart
                        //RUN_BENCHMARK(timers["reinit"], est, estOut);
					
			//Forward problem
			RUN_BENCHMARK(timers["forward"], forward , af::convolve2(est, psf, af::convMode::AF_CONV_DEFAULT));

			//Clamp
			RUN_BENCHMARK(timers["clamp1"], denom , af::max(minVal, forward)); //max necessary since af may choose to use fft conv, which can create negative values

			//ratio
			RUN_BENCHMARK(timers["ratio"], ratio, measured / denom);
			
			//Clamp lower
			RUN_BENCHMARK(timers["clamp2"], ratio , af::min(ratioMax, ratio));

			//Clamp upper
			RUN_BENCHMARK(timers["clamp3"], ratio , af::max(ratioMin, ratio));

			//Compute update
			RUN_BENCHMARK(timers["update"], update , af::convolve2(ratio, psfUD, af::convMode::AF_CONV_DEFAULT));
			
			//Apply update
			RUN_BENCHMARK(timers["mply"], est, update * est);

			//Save intermediate data if specified
			if (rollingSave.size() > 1){
				est.host(outputImage.getMem().get());
				saveEst.writeData(outputImage);
			}


		}

		af::sync();

		cout << "Average iteration time : " << af::timer::stop(loopTime) / iterations * 1000.0 << " [ms]" << endl;

		//Save simulation input and recovered output
		measured.host(simulatedImage.getMem().get());
		est.host(outputImage.getMem().get());

#ifdef BENCHMARK
		for (auto & i : timers){
			cout << i.first << " [ms] = " << i.second * 1000.0 / iterations << endl;

		}

#endif

	}
	catch (af::exception e){

		cout << "Caught af exception while running iterations : " << e.what() << endl;
		return -1;

	}

	return 0;

}

int main(int argc, _TCHAR* argv[])
{

	string filePath;
	int dimXY;
	int deviceID;
	int psfWidth;
	float psfSigma;
	int iterations;
	bool dumpIterations;

	try{

		int argIdx = 1;

		filePath = getArgument(argc, argv, argIdx++);
		dimXY = atoi(getArgument(argc, argv, argIdx++).c_str());
		deviceID = atoi(getArgument(argc, argv, argIdx++).c_str());
		psfWidth = atoi(getArgument(argc, argv, argIdx++).c_str());
		psfSigma = atof(getArgument(argc, argv, argIdx++).c_str());
		iterations = atoi(getArgument(argc, argv, argIdx++).c_str());
		dumpIterations = atoi(getArgument(argc, argv, argIdx++).c_str()) != 0;

	}
	catch (int e){
		cout << "Usage: <filePath> <dimXY> <deviceID> <psfWidth> <psfSigma> <iterations> <dumpIterations>" << endl;
		cout << "Error when converting argument " << e << endl;
		return -1;
	}


	try{
		af::setBackend(af::Backend::AF_BACKEND_OPENCL);
		af::setDevice(deviceID);
		af::info();

		MemStruct<float> image, psf;

		int err = loadInputData(filePath, dimXY, image);

		if (err != 0){
			cout << "Got error " << err << " when reading from file " << filePath << endl;
			return -1;
		}

		err = generatePSF(psfWidth, psfSigma, psf);

		if (err != 0){
			cout << "Got error " << err << " when generating PSF" << endl;
			return -1;
		}

		err = saveDataToFile(filePath + ".psf", psf);
		err |= saveDataToFile(filePath + ".original", image);

		if (err != 0){
			cout << "Error when saving input files: " << err << endl;
			return -1;
		}

		MemStruct<float> sim, out;

		string dumpIntermediates = "";
		if (dumpIterations)
			dumpIntermediates = filePath + ".intermediates";

		err = runRL(psf, image, sim, out, iterations, dumpIntermediates);

		if (err != 0){
			cout << "Error when running iterations: " << endl;
			return -1;
		}

		err = saveDataToFile(filePath + ".simulated", sim);
		err |= saveDataToFile(filePath + ".corrected", out);

		if (err != 0){
			cout << "Error when saving output files: " << err << endl;
			return -1;
		}

	}
	catch (af::exception & e){

		cout << e.what() << endl;
		return -1;
	}

	return 0;
}

