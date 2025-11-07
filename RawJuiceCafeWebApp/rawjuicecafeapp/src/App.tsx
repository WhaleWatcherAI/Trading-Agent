import React, { useState, useMemo, useEffect } from 'react';
import { Menu, ShoppingCart, User, X, Plus, Minus, CheckCircle, LogIn, LogOut } from 'lucide-react';

// Firebase imports (Required for Authentication and Persistence)
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged, signOut } from 'firebase/auth';

// Mock data based on the design images
const menuItems = [
  {
    id: 1,
    name: "RAW GREEN",
    tagline: "Nutrient-Rich Boost",
    ingredients: "Kale, Spinach, Ginger, Lemon, Apple",
    color: "bg-green-500", 
    icon: '🌿',
    price: 9.50,
  },
  {
    id: 2,
    name: "ROOTS ROCK",
    tagline: "Earthly Powerhouse",
    ingredients: "Turmeric, Ginger, Carrot, Apple",
    color: "bg-orange-500", 
    icon: '🥕',
    price: 8.75,
  },
  {
    id: 3,
    name: "WATER MELON",
    tagline: "Hydration Splash",
    ingredients: "Watermelon, Black Salt, Lime",
    color: "bg-red-500",
    icon: '🍉',
    price: 7.99,
  },
  {
    id: 4,
    name: "FOREVER YOUNG",
    tagline: "Tropical Anti-Oxidant",
    ingredients: "Pineapple, Watermelon, Lemon",
    color: "bg-pink-500",
    icon: '🍍',
    price: 9.99,
  },
];

// Custom CSS for the illustrative background pattern (Swirls!)
const customStyle = {
  // Use a complex conic gradient to mimic the organic, swirling background texture
  'background': `repeating-conic-gradient(
    from 0deg,
    #fef08a 0% 10%,
    #fcd34d 10% 20%,
    #fef08a 20% 30%
  ), #fef08a`, // Yellow base
  'backgroundSize': '400px 400px', // Larger size for smooth swirls
  'animation': 'background-scroll 60s linear infinite',
};

// Add a keyframes block for the animation (needs to be injected)
const animationKeyframes = `
  @keyframes background-scroll {
    from { background-position: 0 0; }
    to { background-position: 400px 400px; }
  }
`;

// --- Utility Components ---

// Button component matching the vibrant, outlined style with deeper shadow
const ActionButton = ({ onClick, children, className = '', disabled = false }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`
      px-4 py-2 text-sm font-black uppercase tracking-widest
      text-black bg-lime-400 border-4 border-black rounded-lg
      shadow-[6px_6px_0_#000000] transition-all duration-100 ease-out
      ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-lime-500 active:shadow-[2px_2px_0_#000000] active:translate-x-[4px] active:translate-y-[4px]'}
      ${className}
    `}
  >
    {children}
  </button>
);

// Juice Card Component - Now with dramatic tilt and custom background
const JuiceCard = ({ item, onAddToCart }) => (
  <div 
    className={`p-4 bg-white border-4 border-black rounded-3xl 
      shadow-[12px_12px_0_#000000] 
      transform -rotate-2 hover:rotate-0 transition duration-500 ease-out
      overflow-hidden cursor-pointer
    `}
    style={{ transition: 'transform 0.3s' }}
  >
    {/* Inner Card - Solid Color Block and Ingredient Illustration Layer */}
    <div className={`relative p-6 text-white rounded-2xl border-2 border-black ${item.color} overflow-hidden`}>
      {/* Ingredient Icons Layered in the Background (Super-sized and transparent) */}
      <span className="absolute top-0 left-0 text-[12rem] opacity-20 pointer-events-none transform -translate-x-1/4 -translate-y-1/3 rotate-[-15deg]">{item.icon}</span>
      <span className="absolute bottom-0 right-0 text-[12rem] opacity-20 pointer-events-none transform translate-x-1/4 translate-y-1/3 rotate-[15deg]">{item.icon}</span>
      
      {/* Product Title and Tagline */}
      <h3 className="text-5xl font-black tracking-tighter mb-1 relative z-10 uppercase text-white [text-shadow:_3px_3px_0_rgb(0_0_0_/_100%)]">
        {item.name}
      </h3>
      <p className="text-sm font-bold italic relative z-10 text-black bg-white/50 border-t-2 border-black pt-1 mt-1 inline-block pr-2 pl-1 rounded-sm">{item.tagline}</p>
    </div>

    {/* Details and Action */}
    <div className="mt-4">
      <p className="text-xs text-gray-700 h-8 line-clamp-2 font-medium">{item.ingredients}</p>
      <div className="flex items-center justify-between mt-3">
        {/* Price Tag with a "Splatter" Style Border */}
        <span 
          className="text-4xl font-black text-black bg-lime-100 p-1 px-3 border-4 border-black transform rotate-1"
          style={{ clipPath: 'polygon(0% 15%, 15% 0%, 100% 0%, 100% 85%, 85% 100%, 0% 100%)' }} // Angled/splatter shape
        >
          ${item.price.toFixed(2)}
        </span>
        <ActionButton onClick={() => onAddToCart(item)} className="text-xs !px-4 !py-1">
          GIMME
        </ActionButton>
      </div>
    </div>
  </div>
);


// --- Main Page Views ---

const MenuView = ({ onAddToCart }) => (
  <div className="p-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-10 pt-8" style={customStyle}>
    {menuItems.map(item => (
      <JuiceCard key={item.id} item={item} onAddToCart={onAddToCart} />
    ))}
  </div>
);

const CartView = ({ cart, updateQuantity, removeFromCart, checkout }) => {
  const total = useMemo(() => 
    cart.reduce((sum, item) => sum + item.quantity * item.quantity, 0), 
    [cart]
  );

  if (cart.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 h-96">
        <ShoppingCart className="w-20 h-20 text-gray-400 mb-4 stroke-[4px]" />
        <p className="text-2xl font-black text-black uppercase">Your basket is EMPTY!</p>
        <p className="text-gray-700 font-semibold mt-2">Go grab a RAW BOOST, now!</p>
      </div>
    );
  }

  return (
    <div className="p-4 pt-6">
      <div className="space-y-4">
        {cart.map(item => (
          <div key={item.id} className="flex items-center bg-white p-4 rounded-xl shadow-md border-4 border-black shadow-[4px_4px_0_#000000] transition duration-200 hover:bg-gray-50">
            <div className={`w-14 h-14 flex items-center justify-center rounded-lg text-3xl text-black mr-3 border-4 border-black ${item.color} shadow-[2px_2px_0_#000000]`}>
              {item.icon}
            </div>
            <div className="flex-grow">
              <p className="font-extrabold text-black uppercase">{item.name}</p>
              <p className="text-sm text-gray-600 font-bold">${item.price.toFixed(2)} / ea</p>
            </div>
            
            <div className="flex items-center space-x-2 border-4 border-black rounded-full p-0.5 bg-gray-100 shadow-[2px_2px_0_#000000]">
              <button onClick={() => updateQuantity(item.id, item.quantity - 1)} className="p-1 rounded-full bg-black hover:bg-gray-800 transition"><Minus className="w-4 h-4 text-lime-400" /></button>
              <span className="font-black w-4 text-center text-black">{item.quantity}</span>
              <button onClick={() => updateQuantity(item.id, item.quantity + 1)} className="p-1 rounded-full bg-black hover:bg-gray-800 transition"><Plus className="w-4 h-4 text-lime-400" /></button>
            </div>
            
            <div className="ml-4 w-20 text-right">
              <p className="text-xl font-extrabold text-black">${(item.price * item.quantity).toFixed(2)}</p>
            </div>
            <button onClick={() => removeFromCart(item.id)} className="ml-3 text-red-600 hover:text-red-800 transition">
                <X className="w-6 h-6 stroke-[3px]" />
            </button>
          </div>
        ))}
      </div>

      <div className="mt-8 p-6 bg-lime-200 rounded-2xl border-4 border-black shadow-[10px_10px_0_#000000] border-dashed">
        <div className="flex justify-between items-center text-3xl font-black mb-4 uppercase">
          <span>Total:</span>
          <span className="text-green-700">${total.toFixed(2)}</span>
        </div>
        <ActionButton onClick={checkout} className="w-full text-xl !py-3">
          I'M READY TO PAY!
        </ActionButton>
      </div>
    </div>
  );
};

// Custom Checkout Success Modal - Styled like a burst
const CheckoutModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex items-center justify-center p-4">
      <div 
        className="bg-red-500 p-10 rounded-full border-8 border-black text-center shadow-[15px_15px_0_#000000] w-full max-w-sm transform rotate-[-3deg] active:rotate-0"
        style={{ clipPath: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)' }}
      >
        <CheckCircle className="w-16 h-16 mx-auto text-yellow-300 stroke-[3px] mb-4 animate-bounce" />
        <h2 className="text-4xl font-black text-white uppercase tracking-tighter [text-shadow:_3px_3px_0_rgb(0_0_0_/_100%)]">BAM! Order Up!</h2>
        <p className="mt-2 text-white font-semibold">Your juice is being prepared with RAW energy.</p>
        <ActionButton onClick={onClose} className="mt-6 !bg-yellow-300 !text-black !border-black">
          AWESOME
        </ActionButton>
      </div>
    </div>
  );
};

// --- New Authentication View ---
const AuthView = ({ auth }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMessage, setErrorMessage] = useState('');

    // This is just a visual placeholder since we can't implement real email/password auth 
    // within the Canvas security rules. We use the sign-in options below.
    const handleSubmit = (e) => {
        e.preventDefault();
        setErrorMessage(`Email/password auth is simulated: ${isLogin ? 'Logging In' : 'Signing Up'} for ${email}...`);
        setTimeout(() => setErrorMessage(''), 3000);
    };
    
    // Function to handle logging out, uses the injected Firebase auth instance
    const handleSignOut = async () => {
        try {
            await signOut(auth);
            console.log("User signed out successfully.");
        } catch (error) {
            console.error("Logout failed:", error);
        }
    };

    return (
        <div className="p-8 flex flex-col items-center justify-center min-h-[calc(100vh-120px)]" style={customStyle}>
            
            <div 
                className="w-full max-w-sm bg-white p-6 rounded-3xl border-4 border-black shadow-[15px_15px_0_#000000] transform rotate-1"
            >
                <div className="text-center mb-6">
                    <LogIn className="w-10 h-10 text-black stroke-[3px] mx-auto mb-2" />
                    <h2 className="text-3xl font-black text-black uppercase tracking-tighter">
                        {isLogin ? 'GET YOUR BOOST' : 'JOIN THE RAWNESS'}
                    </h2>
                </div>
                
                {/* Form Placeholder */}
                <form onSubmit={handleSubmit} className="space-y-4">
                    <input
                        type="email"
                        placeholder="EMAIL (Visual Demo Only)"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full p-3 border-4 border-black rounded-lg font-bold placeholder-black/60 bg-yellow-100/80 shadow-[3px_3px_0_#000000]"
                        required
                    />
                    <input
                        type="password"
                        placeholder="PASSWORD (Visual Demo Only)"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full p-3 border-4 border-black rounded-lg font-bold placeholder-black/60 bg-yellow-100/80 shadow-[3px_3px_0_#000000]"
                        required
                    />
                    
                    <ActionButton type="submit" className="w-full text-lg">
                        {isLogin ? 'LOGIN NOW!' : 'SIGN UP!'}
                    </ActionButton>
                </form>

                {errorMessage && (
                    <p className="mt-4 p-2 text-center text-sm font-bold bg-red-500 text-white rounded-lg border-2 border-black">
                        {errorMessage}
                    </p>
                )}

                <div className="mt-6 text-center">
                    <button 
                        onClick={() => setIsLogin(!isLogin)}
                        className="text-sm font-bold text-black underline hover:text-red-600 transition"
                    >
                        {isLogin ? 'Need an account? Sign Up!' : 'Already a member? Log In!'}
                    </button>
                    
                    <div className="border-t-2 border-dashed border-gray-400 mt-4 pt-4">
                        <p className="text-xs font-semibold text-gray-700 uppercase mb-2">
                            OR CONTINUE AS
                        </p>
                        <ActionButton onClick={handleSignOut} className="!bg-red-400 !text-black !text-xs">
                            Guest Mode (Sign Out)
                        </ActionButton>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Profile View - Now shows user data and logout option
const ProfileView = ({ auth, userId }) => {
    
    // Function to handle logging out
    const handleSignOut = async () => {
        try {
            await signOut(auth);
            console.log("User signed out successfully.");
        } catch (error) {
            console.error("Logout failed:", error);
        }
    };

    return (
        <div className="p-8 text-center pt-16">
            <User className="w-20 h-20 text-black stroke-[4px] mx-auto mb-4" />
            <h2 className="text-4xl font-black text-black uppercase tracking-tighter">Your RAW Account</h2>
            <div className="mt-6 p-4 bg-white rounded-lg border-4 border-black shadow-[6px_6px_0_#333333] break-all">
                <p className="text-xs font-semibold text-gray-600 uppercase">Current User ID (Full string):</p>
                <p className="font-mono text-lg font-bold text-black mt-1">{userId}</p>
            </div>
            
            <div className="mt-8">
                <ActionButton onClick={handleSignOut} className="!bg-red-500 !text-white !border-black text-lg">
                    <LogOut className="w-5 h-5 mr-2 inline-block" />
                    LOG OUT
                </ActionButton>
            </div>
        </div>
    );
};


// --- Main App Component ---

const App = () => {
  const [activeTab, setActiveTab] = useState('menu'); // 'menu', 'cart', 'profile'
  const [cart, setCart] = useState([]);
  const [showNotification, setShowNotification] = useState(false);
  const [isCheckoutModalOpen, setIsCheckoutModalOpen] = useState(false);
  
  // Auth State
  const [userId, setUserId] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [auth, setAuth] = useState(null);


  // 1. Firebase Initialization and Auth Listener
  useEffect(() => {
      try {
          const firebaseConfig = JSON.parse(typeof __firebase_config !== 'undefined' ? __firebase_config : '{}');
          // Suppress warnings in console
          // setLogLevel('Debug');
          
          const app = initializeApp(firebaseConfig);
          const authInstance = getAuth(app);
          setAuth(authInstance);

          const unsubscribe = onAuthStateChanged(authInstance, async (user) => {
              if (user) {
                  setUserId(user.uid);
              } else {
                  const token = typeof __initial_auth_token !== 'undefined' ? __initial_auth_token : null;
                  if (token) {
                    // Use the provided custom token
                    await signInWithCustomToken(authInstance, token);
                  } else {
                    // Fallback to anonymous sign-in if no token is available
                    await signInAnonymously(authInstance);
                  }
                  setUserId(authInstance.currentUser?.uid || null);
              }
              setIsAuthReady(true);
          });

          return () => unsubscribe();
      } catch (error) {
          console.error("Firebase initialization failed:", error);
      }
  }, []);


  const totalItems = useMemo(() => 
    cart.reduce((sum, item) => sum + item.quantity, 0), 
    [cart]
  );

  const handleAddToCart = (item) => {
    if (!userId) {
      console.log("Cannot add to cart: User is not authenticated.");
      // Optional: Add a visual alert here instead of using console.log
      return; 
    }
    setCart(prevCart => {
      const existingItem = prevCart.find(i => i.id === item.id);
      if (existingItem) {
        return prevCart.map(i =>
          i.id === item.id ? { ...i, quantity: i.quantity + 1 } : i
        );
      }
      return [...prevCart, { ...item, quantity: 1 }];
    });
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 2000);
  };

  const handleUpdateQuantity = (id, newQuantity) => {
    if (newQuantity <= 0) {
      handleRemoveFromCart(id);
      return;
    }
    setCart(prevCart =>
      prevCart.map(item =>
        item.id === id ? { ...item, quantity: newQuantity } : item
      )
    );
  };

  const handleRemoveFromCart = (id) => {
    setCart(prevCart => prevCart.filter(item => item.id !== id));
  };

  const handleCheckout = () => {
    setIsCheckoutModalOpen(true);
  };
  
  const closeCheckoutModal = () => {
    setIsCheckoutModalOpen(false);
    // Clear cart only after successful "order"
    setCart([]);
    setActiveTab('menu');
  };

  const renderView = () => {
    if (!isAuthReady) {
        return <div className="p-8 text-center pt-16 font-black text-2xl">LOADING RAW DATA...</div>;
    }

    // If user is not logged in (e.g., in anonymous mode), force them to the AuthView
    if (!userId && activeTab !== 'profile') {
        return <AuthView auth={auth} />;
    }

    switch (activeTab) {
      case 'cart':
        return <CartView 
          cart={cart} 
          updateQuantity={handleUpdateQuantity} 
          removeFromCart={handleRemoveFromCart}
          checkout={handleCheckout}
        />;
      case 'profile':
        // If authenticated, show the profile/logout screen
        return <ProfileView auth={auth} userId={userId} />;
      case 'menu':
      default:
        return <MenuView onAddToCart={handleAddToCart} />;
    }
  };

  // Determine if the user is authenticated to hide the cart/profile tabs if they're forced to log in
  const isUserAuthenticated = !!userId;

  return (
    <div className="min-h-screen bg-yellow-300 flex flex-col font-inter">
      
      {/* Inject animation keyframes into a style tag */}
      <style>
        {animationKeyframes}
      </style>
      
      {/* Header - Chaotic Angle and High Contrast */}
      <header className="sticky top-0 z-20 bg-black p-4 shadow-2xl border-b-4 border-lime-400">
        <div className="max-w-xl mx-auto flex justify-between items-center">
          <h1 className="text-4xl font-black text-lime-400 tracking-widest uppercase [text-shadow:_3px_3px_0_rgb(255_255_255_/_100%)]">
            RAW CAFE
          </h1>
          {/* Cart Button only visible when authenticated or on the cart view */}
          {isUserAuthenticated && (
            <button 
                onClick={() => setActiveTab('cart')}
                className="relative p-3 rounded-lg text-black bg-white border-4 border-lime-400 shadow-[4px_4px_0_#333333] hover:bg-gray-100 transition"
            >
                <ShoppingCart className="w-6 h-6 stroke-[3px]" />
                {totalItems > 0 && (
                <span className="absolute -top-1 -right-1 flex items-center justify-center w-5 h-5 text-xs font-black bg-red-600 rounded-full text-white border-2 border-black">
                    {totalItems}
                </span>
                )}
            </button>
          )}
        </div>
      </header>
      
      {/* Main Content Area */}
      <main className="flex-grow max-w-xl w-full mx-auto pb-20">
        {renderView()}
      </main>

      {/* Floating Notification - Styled like a comic burst */}
      <div 
        className={`fixed top-16 right-4 p-4 rounded-lg border-4 border-black shadow-[6px_6px_0_#000000] bg-red-500 text-white font-black transition-all duration-300 z-30 transform -rotate-3
          ${showNotification ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}
          `}
          style={{ clipPath: 'polygon(50% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%, 25% 25%, 75% 25%, 75% 75%, 25% 75%)', width: '150px', textAlign: 'center' }}
      >
        <p className="text-sm">BOTTLE</p>
        <p className="text-lg">ADDED!</p>
      </div>
      
      {/* Bottom Navigation - Fixed, high contrast, and simplified to maximize screen space */}
      <nav className="fixed bottom-0 left-0 right-0 z-20 bg-lime-400 border-t-4 border-black shadow-[0_-8px_0_#000000]">
        <div className="max-w-xl mx-auto flex justify-around">
          {[
            { id: 'menu', Icon: Menu, label: 'MENU' },
            ...(isUserAuthenticated ? [
                { id: 'cart', Icon: ShoppingCart, label: 'CART', count: totalItems },
                { id: 'profile', Icon: User, label: 'ACCOUNT' }
            ] : [
                { id: 'profile', Icon: LogIn, label: 'LOGIN' }
            ]),
          ].map(({ id, Icon, label, count }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex flex-col items-center justify-center py-3 w-full transition-colors 
                ${activeTab === id ? 'text-black font-black bg-lime-500' : 'text-gray-800 hover:bg-lime-300'}
              `}
            >
              <div className="relative">
                <Icon className="w-6 h-6 stroke-[3px]" />
                {count > 0 && id === 'cart' && (
                  <span className="absolute -top-1 -right-1 flex items-center justify-center w-4 h-4 text-xs font-black bg-red-600 rounded-full text-white border-2 border-black">
                    {count}
                  </span>
                )}
              </div>
              <span className="text-xs mt-1 uppercase tracking-wider">{label}</span>
            </button>
          ))}
        </div>
      </nav>
      
      {/* The Custom Checkout Modal */}
      <CheckoutModal isOpen={isCheckoutModalOpen} onClose={closeCheckoutModal} />
      
      {/* Tailwind script import is required for standalone execution */}
      <script src="https://cdn.tailwindcss.com"></script>
    </div>
  );
};

export default App;
