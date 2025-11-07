
import SwiftUI

struct RewardsView: View {
    @State private var points = 120 // Placeholder points

    var body: some View {
        NavigationView {
            VStack {
                Text("Your Points Oasis")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding()

                Text("\(points) Points")
                    .font(.system(size: 60, weight: .bold, design: .rounded))
                    .foregroundColor(Color(red: 0.0, green: 0.5, blue: 0.5)) // Teal color

                Spacer()

                Text("Available Rewards")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                List {
                    Text("Free Smoothie (150 Points)")
                    Text("50% Off Any Bowl (100 Points)")
                    Text("Free Juice (80 Points)")
                }

            }
            .navigationTitle("Rewards")
        }
    }
}

struct RewardsView_Previews: PreviewProvider {
    static var previews: some View {
        RewardsView()
    }
}
