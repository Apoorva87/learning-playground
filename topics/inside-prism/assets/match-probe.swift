import Foundation

// Compile this driver with Prism's original TextMatch.swift.
@main
struct MatchProbe {
    struct Result: Encodable {
        let query: String
        let candidate: String
        let queryWords: [String]
        let candidateWords: [String]
        let tier: String
    }

    static func main() throws {
        let candidate = "Interstellar"
        let candidateWords = TextMatch.tokens(candidate)
        let results = ["in", "INTERSTELLAR", "dune"].map { query in
            let words = TextMatch.tokens(query)
            let match = MatchTier.of(query: words, candidate: candidateWords)
            return Result(query: query, candidate: candidate, queryWords: words,
                          candidateWords: candidateWords, tier: match.map { String(describing: $0) } ?? "none")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(results), as: UTF8.self))
    }
}
