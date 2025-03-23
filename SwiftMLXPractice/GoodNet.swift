//
//  GoodNet.swift
//  SwiftMLXPractice
//
//  Created by 楊敦富 on 2025/3/22.
//

import MLX
import MLXNN
import MLXRandom
import MLXOptimizers
import Foundation
import Gzip
class GoodNet: Module, UnaryLayer {
    
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var pool1: MaxPool2d
    @ModuleInfo var pool2: MaxPool2d
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear
    
    override init() {
        conv1 = Conv2d(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 2)
        conv2 = Conv2d(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0)
        pool1 = MaxPool2d(kernelSize: 2, stride: 2)
        pool2 = MaxPool2d(kernelSize: 2, stride: 2)
        fc1 = Linear(16 * 5 * 5, 120)
        fc2 = Linear(120, 84)
        fc3 = Linear(84, 10)
    }
    
    func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        var x = x
        x = pool1(tanh(conv1(x)))
        x = pool2(tanh(conv2(x)))
        x = flattened(x, start: 1)
        x = tanh(fc1(x))
        x = tanh(fc2(x))
        x = fc3(x)
        return x
    }
}

fileprivate func getGoodNetLossFunction(model: GoodNet, x: MLXArray, y: MLXArray) -> MLXArray {
    crossEntropy(logits: model(x), targets: y, reduction: .mean)
}

fileprivate func getGoodNetEvalFunction(model: GoodNet, x: MLXArray, y: MLXArray) -> MLXArray {
    mean(argMax(model(x), axis: 1) .== y)
}

func mnistExample() async throws {
    // download data
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
    let weightUrl = url.appending(component: "weights.safetensors")
    
    
    try await download(into: url)
    let data = try load(from: url)

    let trainImages = data[.init(.training, .images)]!
    let trainLabels = data[.init(.training, .labels)]!
    let testImages = data[.init(.test, .images)]!
    let testLabels = data[.init(.test, .labels)]!
    
    let model = GoodNet()
    if  FileManager.default.fileExists(atPath: weightUrl.path()) {
        let weights = try ModuleParameters.unflattened(loadArrays(url: weightUrl))
        try model.update(parameters: weights, verify: .noUnusedKeys)
    }
    
    eval(model)
    MLXRandom.seed(0)
    var generator: RandomNumberGenerator = SplitMix64(seed: 0)
    let lg = valueAndGrad(model: model, getGoodNetLossFunction)
    let optimizer = SGD(learningRate: 1e-1)
    
    func step(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let (loss, grads) = lg(model, x, y)
        optimizer.update(model: model, gradients: grads)
        return loss
    }
    
    let resolvedStep = MLX.compile(inputs: [model, optimizer], outputs: [model, optimizer], step)
    
    for e in 0..<30 {
        let start = Date.timeIntervalSinceReferenceDate
        for (x, y) in iterateBatches(batchSize: 256, x: trainImages, y: trainLabels, using: &generator) {
            _ = resolvedStep(x, y)
            eval(model, optimizer)
        }
        
        let accuracy = getGoodNetEvalFunction(model: model, x: testImages, y: testLabels)
        let end = Date.timeIntervalSinceReferenceDate
        print(
            """
            Epoch \(e): test accuracy \(accuracy.item(Float.self).formatted())
            Time: \((end - start).formatted())

            """
        )
    }
    
    
    // save weight
    let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    try save(arrays: weights, url: weightUrl)
    print("Saved weight to", weightUrl)
}

fileprivate struct BatchSequence: Sequence, IteratorProtocol {
    typealias Element = (MLXArray, MLXArray)
    let batchSize: Int
    let x: MLXArray
    let y: MLXArray
    
    let indexes: MLXArray
    var index = 0
    
    init(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator) {
        self.batchSize = batchSize
        self.x = x
        self.y = y
        let a = Array(0..<y.size).shuffled(using: &generator)
        self.indexes = MLXArray(a)
    }
    
    mutating func next() -> Element? {
        guard index < y.size else { return nil }
        
        let range = index ..< Swift.min(index + batchSize, y.size)
        index += batchSize
        let ids = indexes[range]
        return (x[ids], y[ids])
    }
}

fileprivate func iterateBatches(
    batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator
) -> some Sequence<(MLXArray, MLXArray)> {
    BatchSequence(batchSize: batchSize, x: x, y: y, using: &generator)
}


fileprivate enum Use: String, Hashable, Sendable {
    case test
    case training
}

fileprivate enum DataKind: String, Hashable, Sendable {
    case images
    case labels
}

fileprivate struct FileKind: Hashable, CustomStringConvertible, Sendable {
    let use: Use
    let data: DataKind

    public init(_ use: Use, _ data: DataKind) {
        self.use = use
        self.data = data
    }

    public var description: String {
        "\(use.rawValue)-\(data.rawValue)"
    }
}

fileprivate struct LoadInfo: Sendable {
    let name: String
    let offset: Int
    let convert: @Sendable (MLXArray) -> MLXArray
}

fileprivate let files = [
    FileKind(.training, .images): LoadInfo(
        name: "train-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.test, .images): LoadInfo(
        name: "t10k-images-idx3-ubyte.gz",
        offset: 16,
        convert: {
            $0.reshaped([-1, 28, 28, 1]).asType(.float32) / 255.0
        }),
    FileKind(.training, .labels): LoadInfo(
        name: "train-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
    FileKind(.test, .labels): LoadInfo(
        name: "t10k-labels-idx1-ubyte.gz",
        offset: 8,
        convert: {
            $0.asType(.uint32)
        }),
]


fileprivate func download(into: URL) async throws {
    for (_, info) in files {
        let fileURL = into.appending(component: info.name)
        let baseURL = URL(string: "https://raw.githubusercontent.com/fgnt/mnist/master/")!
        
        if !FileManager.default.fileExists(atPath: fileURL.path()) {
            print("Download: \(info.name)")
            
            let url = baseURL.appending(component: info.name)
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse else {
                fatalError("Unable to download \(url), not an http response: \(response)")
            }
            guard httpResponse.statusCode == 200 else {
                fatalError("Unable to download \(url): \(httpResponse)")
            }

            try data.write(to: fileURL)
        }
    }
}

fileprivate func load(from: URL) throws -> [FileKind: MLXArray] {
    var result = [FileKind: MLXArray]()

    for (key, info) in files {
        let fileURL = from.appending(component: info.name)
        let data = try Data(contentsOf: fileURL).gunzipped()

        let array = MLXArray(
            data.dropFirst(info.offset), [data.count - info.offset], type: UInt8.self)

        result[key] = info.convert(array)
    }

    return result
}

// From https://github.com/apple/swift/blob/cb0fb1ea051631219c0b944b84c78571448d58c2/benchmark/utils/TestsUtils.swift#L254
//
// This is just a seedable RandomNumberGenerator for shuffle()

// This is a fixed-increment version of Java 8's SplittableRandom generator.
// It is a very fast generator passing BigCrush, with 64 bits of state.
// See http://dx.doi.org/10.1145/2714064.2660195 and
// http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
//
// Derived from public domain C implementation by Sebastiano Vigna
// See http://xoshiro.di.unimi.it/splitmix64.c
fileprivate struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        self.state &+= 0x9e37_79b9_7f4a_7c15
        var z: UInt64 = self.state
        z = (z ^ (z &>> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z &>> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z &>> 31)
    }
}
