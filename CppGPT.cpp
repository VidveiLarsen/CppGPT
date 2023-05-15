#include "CppGPT.h"
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <string_view>
#include <string>
#include <filesystem>

using EncodedString = std::vector<int32_t>;

class TextEncoder
{
public:
	virtual void GenerateEncodings(const std::string_view data) = 0;
	// string, or uint32_t/uint16_t as return?
	virtual EncodedString Encode(const std::string_view data) const = 0;
	virtual std::string Decode(const EncodedString& data) const = 0;
};

class CharEncoder : public TextEncoder
{
	std::vector<uint16_t> m_encodings;
	std::vector<char> m_inverseEncodings;
public:
	void GenerateEncodings(const std::string_view data) override
	{
		std::set<char> chars{};
		for (const char letter : data)
		{
			chars.insert(letter);
		}

		m_encodings.resize(1024);
		
		uint16_t encodingIndex = 0;
		for (const char letter : chars)
		{
			m_encodings.at(static_cast<uint16_t>(letter)) = encodingIndex;
			encodingIndex++;
			m_inverseEncodings.push_back(letter);
		}
	}
	
	EncodedString Encode(const std::string_view data) const override
	{
		EncodedString outData{};
		outData.reserve(data.size());

		for (const char letter : data )
		{
			const char encoded = m_encodings.at(static_cast<uint16_t>(letter));
			outData.push_back(encoded);
		}
		return outData;
	}

	std::string Decode(const EncodedString& data) const 
	{
		std::string outData{};
		outData.reserve(data.size());

		for (const auto dataPoint : data)
		{
			outData.push_back(m_inverseEncodings.at(dataPoint));
		}
		return outData;
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
	const uint16_t _contextSize;
public:
	TextDataSet(const std::filesystem::path& filePath, TextEncoder& encoder, const uint16_t contextSize):
	_encoder(encoder), _contextSize(contextSize)
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
		return c10::optional<size_t>(_encodedData.size() - _contextSize);
	}

  	ExampleType get(size_t index) override
	{
		// up to size, which is smaller than _encodedData.size(). index is from end
		at::Tensor target = torch::from_blob(&_encodedData.at(index + _contextSize), {1}, at::kInt);
		at::Tensor data = torch::from_blob(&_encodedData.at(index), {_contextSize}, at::kInt);
		return torch::data::Example(data, target);
	}
};

class CppGPT : torch::nn::Module
{
public:
	CppGPT()
	{

	}
	

};

int main()
{
	try
	{
		CharEncoder encoder{};
		TextDataSet dataset{"F:\\CppGPT\\Data\\input.txt", encoder, 10};

	
		// define model
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}

	return 0;
}
