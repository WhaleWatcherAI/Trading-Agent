
import Foundation

struct Reward: Identifiable, Codable {
    let id: UUID
    let name: String
    let description: String
    let pointsRequired: Int
}
