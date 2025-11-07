
import Foundation

class MenuViewModel: ObservableObject {
    @Published var menuItems: [MenuItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String? = nil

    private let menuURL = "http://localhost:3003/api/menu"

    func fetchMenu() {
        isLoading = true
        errorMessage = nil

        guard let url = URL(string: menuURL) else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }

        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                self.isLoading = false
                if let error = error {
                    self.errorMessage = error.localizedDescription
                    return
                }

                guard let data = data else {
                    self.errorMessage = "No data received"
                    return
                }

                do {
                    self.menuItems = try JSONDecoder().decode([MenuItem].self, from: data)
                } catch {
                    self.errorMessage = "Failed to decode menu: \(error.localizedDescription)"
                    print("Decoding Error: \(error)")
                }
            }
        }.resume()
    }
}
