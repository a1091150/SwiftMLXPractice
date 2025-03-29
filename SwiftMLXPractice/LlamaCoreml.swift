//
//  LlamaCoreml.swift
//  SwiftMLXPractice
//
//  Created by 楊敦富 on 2025/3/27.
//

import Hub
import Tokenizers
import Jinja
import CoreML
import Models
import Generation

fileprivate let baseUrl: URL = {
    return URL.homeDirectory
        .appending(path: ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
                   directoryHint: .isDirectory)
}()

fileprivate func readConfigJson() throws -> Config?{
    
    let path = baseUrl.appending(path: "tokenizer_config.json", directoryHint: .notDirectory)
    let data = try Data(contentsOf: path)
    if  let configJson = try JSONSerialization.jsonObject(with: data) as? [NSString: Any] {
        return Config(configJson)
    }
    return nil
}

fileprivate func readTokenizerData() throws -> Config? {
    let path = baseUrl.appending(path: "tokenizer.json", directoryHint: .notDirectory)
    let data = try Data(contentsOf: path)
    if  let configJson = try JSONSerialization.jsonObject(with: data) as? [NSString: Any] {
        return Config(configJson)
    }
    return nil
}

fileprivate func readMLModel() throws -> MLModel {
    let path = URL.documentsDirectory.appending(
        path: "mlx_project/llama-to-coreml/model.mlmodelc",
        directoryHint: .notDirectory)
    
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    let model = try MLModel(contentsOf: path, configuration: config)
    return model
}

fileprivate enum Keys {
    // Input keys
    static let inputIds = "inputIds"
    static let attentionMask = "attentionMask"
    static let causalMask = "causalMask"
    static let keyCache = "keyCache"
    static let valueCache = "valueCache"
    // Output keys
    static let logits = "logits"
    static let presentKeys = "presentKeys"
    static let presentValues = "presentValues"
}

fileprivate func predictNextTokenScores(
    tokens: MLTensor,
    config: GenerationConfig,
    model: MLModel,
    maxContextLength: Int = 128,
    isRequiringAttentionMask: Bool = true,
    state: MLState
) async throws -> MLTensor {
    assert(tokens.rank == 2) // [batch, current sequence length]
    let tokenCount = tokens.shape[1]
    let inputIds = tokens
    let isRequiringCausalMask = true
    
    var inputDictionary = [
        Keys.inputIds: inputIds
    ]
    
    
    if  isRequiringCausalMask {
        let causalMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
        inputDictionary[Keys.causalMask] = causalMask
    }
    
    let outputs = try await model.prediction(from: inputDictionary, using: state)
    
    assert(outputs.keys.contains(Keys.logits))
    let scores = outputs[Keys.logits]!
    assert(scores.rank == 3)
    
    let tokenIndex = inputIds.shape[1] - 1
    let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)
    assert(nextTokenScores.rank == 3)
    assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
    return nextTokenScores
}

func llamaRun() async throws{
    let tConfig = try readConfigJson()!
    let tData = try readTokenizerData()!
    let tokenizer = try PreTrainedTokenizer(tokenizerConfig: tConfig, tokenizerData: tData)
    let tokens = tokenizer.encode(text: "介紹你自己").map{ Int32($0) }
    
    var config = GenerationConfig(maxNewTokens: 128)
    config.eosTokenId = tokenizer.eosTokenId
    config.bosTokenId = tokenizer.bosTokenId
    config.maxLength = config.maxNewTokens + 128
    
    
    let model = try readMLModel()
    // copied from LanguageModelWithStatefulKVCache
    
    let state = model.makeState()
    
    // Copy from Generation.swift generation function
    let batchTokens = tokens.map{$0}
    var outputTokens = MLTensor(batchTokens).expandingShape(at: 0)
    while outputTokens.shape[1] < config.maxLength {
        let nextTokenScores = try await predictNextTokenScores(tokens: outputTokens, config: config,
                                                               model: model, state: state)
        let nextToken = switch config.generationMode {
        case .greedy:
            selectNextTokenUsingGreedyDecoding(from: nextTokenScores)
        case .sample:
            selectNextTokenUsingTopKSampling(
                from: nextTokenScores,
                temperature: Float(config.temperature),
                topK: config.topK
            )
        default:
            fatalError("Generation mode \(config.generationMode) not implemented yet")
        }
        
        outputTokens = MLTensor(concatenating: [outputTokens, nextToken], alongAxis: -1)
    }
    
    let outputTokenInt = await tensorToGenerationOutput(outputTokens)
    let final = tokenizer.decode(tokens: outputTokenInt)
    print(final)
}

private func tensorToGenerationOutput(_ tensor: MLTensor) async -> GenerationOutput {
    await tensor.shapedArray(of: Int32.self).scalars.map { Int($0) }
}

// Copy from Generation Decoder.swift
private func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
}

private func selectNextTokenUsingTopKSampling(from scores: MLTensor, temperature: Float, topK: Int) -> MLTensor {
    let temperatureAdjustedScores = scores / temperature
    let (topKScores, topKIndices) = temperatureAdjustedScores.topK(topK)
    let topKProbs = topKScores.softmax(alongAxis: -1)
    let rnd = topKProbs.sum() * Float.random(in: 0 ..< 1)
    var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
    accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
    let topKIndex = accumTopKProbs.argsort()[..., 0]
    let nextTokenTensor = topKIndices.gathering(
        atIndices: topKIndex,
        alongAxis: topKIndices.rank - 1
    )
    return nextTokenTensor.reshaped(to: [1, 1])
}
