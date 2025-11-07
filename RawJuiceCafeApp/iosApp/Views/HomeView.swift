
import SwiftUI

struct HomeView: View {
    var body: some View {
        NavigationView {
            VStack {
                Text("Welcome to Raw Juice Cafe")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding()

                Text("Your daily dose of sunshine.")
                    .font(.headline)
                    .foregroundColor(.secondary)

                Spacer()

                Button(action: {
                    // Action for ordering
                }) {
                    Text("Order Now")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(width: 220, height: 60)
                        .background(Color(red: 0.0, green: 0.5, blue: 0.5)) // Teal color
                        .cornerRadius(15.0)
                }

                Spacer()
            }
            .navigationTitle("Home")
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}
