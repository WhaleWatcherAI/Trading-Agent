
import Foundation

struct MenuItem: Identifiable, Codable {
    let id: UUID
    let name: String
    let description: String
    let price: Double
    let category: String
    let imageName: String // To hold a placeholder image name
}
