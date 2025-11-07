
import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }

            MenuView()
                .tabItem {
                    Label("Menu", systemImage: "book.fill")
                }

            RewardsView()
                .tabItem {
                    Label("Rewards", systemImage: "star.fill")
                }

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
        }
        .accentColor(Color(red: 0.0, green: 0.5, blue: 0.5)) // A teal color for the "healthy, surftown" vibe
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
