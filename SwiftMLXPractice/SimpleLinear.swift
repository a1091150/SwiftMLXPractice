//
//  SimpleLinear.swift
//  SwiftMLXPractice
//
//  Created by 楊敦富 on 2025/3/23.
//

import MLX
import MLXNN
import Foundation
import MLXRandom
import MLXOptimizers
class LinearFunctionModel: Module, UnaryLayer {
    let m = MLXRandom.uniform(low: -5.0, high: 5.0)
    let b = MLXRandom.uniform(low: -5.0, high: 5.0)


    func callAsFunction(_ x: MLXArray) -> MLXArray {
        m * x + b
    }
}

func simpleLinearExample() {
    let model = LinearFunctionModel()
    eval(model)
    
    let lg = valueAndGrad(model: model, loss)
    let optimizer = SGD(learningRate: 1e-1)
    
    for _ in 0..<30 {
        let b = model.b
        let m = model.m
        print("target: b = \(b), m = \(m)")
        print("parameters: \(model.parameters())")
        
        let x = MLXRandom.uniform(low: -5.0, high: 5, [10, 1])
        let y = f(x)
        eval(x, y)
        
        let (loss, grads) = lg(model, x, y)
        print(loss, "\n")
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)
    }
}

/// Return y, the codomain of x
fileprivate func f(_ x: MLXArray) -> MLXArray {
    let m = 0.25
    let b = 7
    return m * x + b
}

fileprivate func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}
