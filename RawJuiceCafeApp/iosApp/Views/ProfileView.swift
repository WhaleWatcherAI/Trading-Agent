
import SwiftUI

struct ProfileView: View {
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Account")) {
                    NavigationLink(destination: Text("Edit your profile details.")) {
                        Text("Edit Profile")
                    }
                    NavigationLink(destination: Text("View your past orders.")) {
                        Text("Order History")
                    }
                }

                Section(header: Text("Settings")) {
                    NavigationLink(destination: Text("Manage notifications.")) {
                        Text("Notifications")
                    }
                }
                
                Section {
                    Button(action: {
                        // Log out action
                    }) {
                        Text("Log Out")
                            .foregroundColor(.red)
                    }
                }
            }
            .navigationTitle("Profile")
        }
    }
}

struct ProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileView()
    }
}
