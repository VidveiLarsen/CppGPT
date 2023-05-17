#include "CppGPT.h"
#include <torch/torch.h>

#include <random>
#include <algorithm>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <string_view>
#include <string>
#include <filesystem>
#include <limits>
#include <format>

#include "spdlog/spdlog.h"

#define LOGTENSOR(x) \
{ \
	std::stringstream ss; \
	ss << x; \
	SPDLOG_INFO(ss.str()); \
}

using EncodedString = std::vector<int32_t>;

class TextEncoder
{
public:
	virtual void GenerateEncodings(const std::string_view data) = 0;
	virtual uint16_t UniqueTokenCount() const = 0;
	virtual EncodedString Encode(const std::string_view data) const = 0;
	virtual std::string Decode(const EncodedString& data) const = 0;
};

class CharEncoder : public TextEncoder
{
	std::vector<char> m_inverseEncodings;
	std::map<char, uint32_t> m_mapping{};
public:
	void GenerateEncodings(const std::string_view data) override
	{
		for (const char letter : data)
		{
			if (m_mapping.find(letter) == m_mapping.end())
			{
				m_mapping.try_emplace(letter, m_mapping.size());
				m_inverseEncodings.push_back(letter);
			}
		}
	}
	
	EncodedString Encode(const std::string_view data) const override
	{
		EncodedString outData{};
		outData.reserve(data.size());

		for (const char letter : data)
		{
			outData.push_back(m_mapping.at(letter));
		}
		return outData;
	}

	std::string Decode(const EncodedString& data) const override 
	{
		std::string outData{};
		outData.reserve(data.size());

		for (const auto dataPoint : data)
		{
			outData.push_back(m_inverseEncodings.at(dataPoint));
		}
		return outData;
	}

	uint16_t UniqueTokenCount() const override 
	{
		return m_inverseEncodings.size();
	}
};

// TODO
// class BytePairEncoding : public TextEncoder
// {
// 	BytePairEncoding()
// };

class TextDataSet : public torch::data::Dataset<TextDataSet, torch::data::Example<>>
{
	TextEncoder& _encoder;
	EncodedString _encodedData;
	const uint16_t m_contextSize;
public:
	TextDataSet(const std::filesystem::path& filePath, TextEncoder& encoder, const uint16_t contextSize):
	_encoder(encoder), m_contextSize(contextSize)
	{
		if(!std::filesystem::exists(filePath))
		{
			throw std::runtime_error("File does not exist!");
		}

		std::ifstream file{filePath};
		if (!file.good() || !file.is_open())
		{
			throw std::runtime_error("Failed to open " + filePath.string());
		}

		std::stringstream buffer;
		buffer << file.rdbuf();
		
		const std::string fileContent = buffer.str();
		_encoder.GenerateEncodings(fileContent);
		_encodedData = _encoder.Encode(fileContent);

		const std::string reDecodedData = _encoder.Decode(_encodedData);
		if (reDecodedData != fileContent)
		{
			throw std::runtime_error("Encoded + decoded != raw data");
		}
	}
 	
	c10::optional<size_t> size() const override
	{
		return c10::optional<size_t>(_encodedData.size() - m_contextSize - 1);
	}

  	ExampleType get(size_t index) override
	{
		// up to size, which is smaller than _encodedData.size(). index is from end
		at::Tensor data = torch::from_blob(&_encodedData.at(index), {m_contextSize}, at::kInt);

		at::Tensor target = torch::from_blob(&_encodedData.at(index + 1), {m_contextSize}, at::kInt).to(at::kLong);


		return torch::data::Example(data, target);
	}
};

constexpr uint8_t batchSize{64};
constexpr uint8_t contextMaxSize{40};

constexpr uint8_t embeddingDim {128};

constexpr uint8_t keyQueryDim {embeddingDim};
constexpr uint8_t valueDim {embeddingDim};

class SelfAttention : public torch::nn::Module
{
	torch::Tensor m_wKey{nullptr};
	torch::Tensor m_wQuery{nullptr};
	torch::Tensor m_wValue{nullptr};
	torch::DeviceType m_device{torch::kCPU};
	torch::Tensor m_minusInfty{nullptr};
	torch::Tensor m_tril{nullptr};
public:
	SelfAttention(): 
		m_wKey(register_parameter("m_wKey",torch::rand({keyQueryDim, embeddingDim}).sub(0.5f).mul(2).div(std::sqrt(float(embeddingDim))), true)),
		m_wQuery(register_parameter("m_wQuery",torch::rand({keyQueryDim, embeddingDim}).sub(0.5f).mul(2).div(std::sqrt(float(embeddingDim))), true)),
		m_wValue(register_parameter("m_wValue",torch::rand({valueDim, embeddingDim}).sub(0.5f).mul(2).div(std::sqrt(float(embeddingDim))), true))		
	{
		auto minusInfty = std::numeric_limits<float>::lowest();
		m_minusInfty = register_buffer("minusInfty", torch::from_blob(&minusInfty, {1}, torch::TensorOptions(torch::kFloat))).squeeze(0);
		
		m_tril = register_buffer("m_tril", torch::tril(torch::ones({contextMaxSize, contextMaxSize})).eq(torch::Scalar(0)).unsqueeze(0));// {1(batch), context, context}
	}
	
	torch::Tensor forward(torch::Tensor x) //{batch, context, embeddingDim}
	{
		const auto context = x.size(1);
		at::Tensor xT = x.transpose(1, 2); // {batch, embeddingDim, context}
		
		at::Tensor key = m_wKey.matmul(xT); // {batch, keyQueryDim, context}
		at::Tensor query = m_wQuery.matmul(xT); // {batch, keyQueryDim, context}
		
		// QT: {T, C} x {C, T} = {T, T}

		at::Tensor weight = query.transpose(1,2).matmul(key); // {batch, context, context}

		auto tril = m_tril.slice(1,0,context).slice(2, 0, context); 
		torch::Tensor weightMasked = weight.masked_fill(tril, m_minusInfty);

		// then zero out. Depends on what the goal is. 
		// encoder does not have this masking, but a decoder does. 

		// weight = // (Q*xT)T*(K*xT) = x*QT*K*xT . Might as well define Q to be transpose initially. 
		// Also, as QT*K has same number of params, as embeddingDim^2, why keep such a high number of params?

		// {context, embeddingDim} x {N, M } x {embeddingTime, context}. H
		// N=M=embeddingDim. ? or simply a linear layer?

		at::Tensor weightNormalized = torch::nn::functional::softmax(weightMasked / std::sqrt(keyQueryDim), torch::nn::functional::SoftmaxFuncOptions(2)); // {context, context}
		
		at::Tensor value = m_wValue.matmul(xT).transpose(1,2); // {batch, context,  valueDim}
		at::Tensor contextVector = weightNormalized.matmul(value); // {context, context } x {context, valueDim} = {context, valueDim} (due to key)
		return contextVector;
	}


  	void to(torch::Device device, bool non_blocking = false) override
	{
		m_device = device.type();
		torch::nn::Module::to(device, non_blocking);
	}
	
};

class CppGPT : public torch::nn::Module
{

	torch::nn::Linear m_lmHead{nullptr};

	
	torch::nn::Embedding m_embedder{nullptr};
	torch::nn::Embedding m_embedderPosition{nullptr};
	std::shared_ptr<SelfAttention> m_att{};

	torch::Tensor m_posRange{nullptr};
public:
	CppGPT(const uint16_t vocabSize)
	{
		m_lmHead = register_module("wlmHead", torch::nn::Linear(torch::nn::LinearOptions(embeddingDim, vocabSize).bias(false)));
		
		m_embedder = register_module("embedder", torch::nn::Embedding(vocabSize, embeddingDim));
		m_embedderPosition = register_module("embedderPosition", torch::nn::Embedding(contextMaxSize, embeddingDim)); // limits the context size
		
		m_att = register_module("m_att", std::make_shared<SelfAttention>());
		m_posRange = register_buffer("m_posRange", torch::arange(contextMaxSize).unsqueeze(0)); // {1, contextMax}
	}

	torch::Tensor attention(const torch::Tensor x) // x {batch, context, embeddingDim}
	{	
		
		torch::Tensor y = m_att->forward(x);
		return y;
		// TODO fan out, relu, fan in (could here specify embedding dim easily)
		// need to get to embeddingDim
		//at::Tensor y1 = torch::nn::functional::relu(y);
		//return y1; // {context, embeddingDim}. Should be!
	}

	torch::Tensor forward(const torch::Tensor x) // x: {batch, context} -> {batch, vocabSize, context}
	{
		const auto c = x.size(1);
		torch::Tensor xEmbedded = m_embedder(x); // {batch, context, embeddingDim}
		torch::Tensor posEmbedding = m_embedderPosition(m_posRange).slice(1, 0, c); //{batch, context, embeddingDim}

		// how is position useful/affects the gradients?
		torch::Tensor xIn = xEmbedded + posEmbedding; // {batch, context, embeddingDim}
		
		at::Tensor y2 = attention(xIn); // {batch, context, embeddingDim}
		
		at::Tensor logits = m_lmHead(y2); // {batch, context, embeddingDim} x {batch, embeddingDim, vocabSize} = {batch, context, vocabSize} // opposite order

		return logits.transpose(1,2);	// {batch, vocabSize, context}
	}

	torch::Tensor loss(torch::Tensor logits, torch::Tensor target) // {batch, vocabSize, context}, {batch, 1}
	{
		// LOGTENSOR(logits.sizes());
		const auto B = logits.size(0);
		const auto V = logits.size(1);
		const auto C = logits.size(2);
		torch::Tensor lg2 = logits.transpose(1,2);
		torch::Tensor logits2 = lg2.view({B*C, V}); // {batch, context, vocabSize } -> {batch*context, vocabSize}
		
		torch::Tensor targets2 = target.view(B*C); // {batch, context} -> {batch * context}

		return torch::nn::functional::cross_entropy(logits2, targets2);
	}

	torch::Tensor generate(torch::Tensor input, const uint16_t steps) // {context} -> {steps}
	{
		// only feed in what we have for now as indices
		// modify model to handle context <= contextSizeMax
		torch::Tensor inputx = input.unsqueeze(0); // {batch, context}
		torch::NoGradGuard noGrad{};
		for (int newTokenId = 0; newTokenId < steps; newTokenId++)
		{
			torch::Tensor x = inputx.size(1) < contextMaxSize ? inputx : inputx.slice(1, -contextMaxSize);
			torch::Tensor logits = forward(x); // {batch, vocabSize, context}
			// const auto vocabSize = logits.size(2);
			torch::Tensor logit2 = logits.slice(2, -1).squeeze(2); // {batch, vocabSize};
			// ignore top K and temperature for now
			torch::Tensor probs = torch::nn::functional::softmax(logit2, 1); // {batch, vocabSize}
			torch::Tensor idx_next = torch::multinomial(probs, 1); // {batch, 1} value is [0, vocabSize-1]
			inputx = torch::cat({inputx, idx_next}, 1); // {batch, context : steps}
		}
		return inputx.squeeze(0); // {steps}
	}
};



int main()
{
	try
	{
		CharEncoder encoder{};
		auto trainSet = TextDataSet("D:\\CppGPT\\data\\train.txt", encoder, contextMaxSize).map(torch::data::transforms::Stack<>());
		auto valSet = TextDataSet("D:\\CppGPT\\data\\val.txt", encoder, contextMaxSize).map(torch::data::transforms::Stack<>());
		auto testSet = TextDataSet("D:\\CppGPT\\data\\test.txt", encoder, contextMaxSize).map(torch::data::transforms::Stack<>());
		
		
		auto dataLoaderTraining = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(trainSet,  torch::data::DataLoaderOptions().batch_size(batchSize));
		auto dataLoaderValidation = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(valSet,  torch::data::DataLoaderOptions().batch_size(batchSize));
		auto dataLoaderTesting = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(testSet,  torch::data::DataLoaderOptions().batch_size(batchSize));
		CppGPT model{encoder.UniqueTokenCount()};
		torch::DeviceType device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		model.to(device);
		const auto trainingSteps = trainSet.size().value() / batchSize;
		torch::optim::AdamW adam{model.parameters(), torch::optim::AdamWOptions(1e-3)}; 
		for (uint32_t epoch = 0; epoch < 10; epoch++)
		{ 
			int step = 0;
			for (auto& batch : *dataLoaderTraining)
			{
				{
					torch::Tensor logits = model.forward(batch.data.to(device)).to(torch::kCPU);
					torch::Tensor loss = model.loss(logits, batch.target);

					step++;
					if (step % 500 == 0)
					{
						std::cout << " epoch " << epoch << "/" << 10 << " - step " <<  step << "/" << trainingSteps << std::endl;
						LOGTENSOR(loss);
					}

					adam.zero_grad();
					loss.backward();
					adam.step();	
				}
				if (step % 500 == 0)
				{
					torch::NoGradGuard noGrad{};
					{
						torch::Tensor lossVals = torch::zeros({1});
						for (auto& batchVal : *dataLoaderValidation)
						{
							torch::Tensor lossVal = model.loss(model.forward(batchVal.data.to(device)), batchVal.target.to(device)).to(torch::kCPU);
							lossVals = torch::cat({lossVals, lossVal.unsqueeze(0)}, 0);
						}
						std::cout << "Validation loss: " << lossVals.mean(0) << std::endl;
					}

					// something very wrong. should not reach such a low val loss, but garbadge generation. 
					
					//torch::save({model.parameters()}, std::format("D:\\CppGPT\\checkpoints\\test_step{}_epoch{}.pth", step, epoch));
					
					const std::string prompt{"Thomas:"};
					EncodedString input = encoder.Encode(prompt);
					torch::Tensor data = torch::from_blob(&input.at(0), {static_cast<long long>(input.size())}, at::kInt).to(device);
					torch::Tensor result = model.generate(data, 100).to(torch::kInt32).to(torch::kCPU); 
					
					auto resultAcc = result.accessor<int32_t, 1>();
					EncodedString resultString{};

					for (int i = 0; i < 100; i++)
					{
						resultString.push_back(static_cast<uint32_t>(resultAcc[i]));
					}
					
					const std::string generatedString = encoder.Decode(resultString);
					std::cout << "Generated string: " << generatedString << std::endl;
				}
			}

		}
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}
