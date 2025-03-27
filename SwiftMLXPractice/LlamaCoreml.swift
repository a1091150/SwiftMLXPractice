//
//  LlamaCoreml.swift
//  SwiftMLXPractice
//
//  Created by 楊敦富 on 2025/3/27.
//

import Hub
import Tokenizers
import Jinja
import MLX
import CoreML

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

func llamaRun() throws{
    let tConfig = try readConfigJson()!
    let tData = try readTokenizerData()!
    let tokenizer = try PreTrainedTokenizer(tokenizerConfig: tConfig, tokenizerData: tData)
    let tokens = tokenizer.encode(text: "123")
    print(tokens)
}
