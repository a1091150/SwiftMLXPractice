//
//  main.swift
//  SwiftMLXPractice
//
//  Created by 楊敦富 on 2025/3/22.
//

import Foundation
import Hub
print("Hello, World!")

//simpleLinearExample()
//try await mnistExample()
do {
    try await llamaRun()
}catch {
    print(error.localizedDescription)
    print(error)
}

