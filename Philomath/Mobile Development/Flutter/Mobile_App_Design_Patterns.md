
# 📱 Mobile App Design Patterns

This document explains key design patterns used in mobile application development. These patterns help you build scalable, maintainable, and user-friendly apps.

---

## 🧱 1. Architectural Patterns

### MVC (Model-View-Controller)
- **Model**: Handles data and business logic.
- **View**: UI elements.
- **Controller**: Handles user input and updates model/view.
- **Used in**: iOS (UIKit), small Android apps.

✅ Easy to implement  
❌ Becomes messy in large apps

---

### MVVM (Model-View-ViewModel)
- **Model**: Data and logic
- **View**: UI
- **ViewModel**: Connects Model and View using data binding
- **Used in**: Android (Jetpack), Flutter (with Provider, Riverpod)

✅ Great for testing and clean UI logic separation  
❌ Slightly more complex for beginners

---

### MVP (Model-View-Presenter)
- **Presenter**: Similar to ViewModel, but it controls the view more directly
- **Used in**: Older Android apps

✅ Testable and modular  
❌ Can lead to large Presenter files

---

## 🔄 2. Behavioral Patterns

### Observer Pattern
- Notifies objects automatically when one object changes.
- Example: UI auto-updates when the cart changes.

✅ Used in: LiveData (Android), Swift Combine, Flutter streams

---

### Command Pattern
- Encapsulates a request as an object (e.g., button press triggers a command)

---

### Strategy Pattern
- Selects algorithm/behavior at runtime.
- Example: Choosing sorting or payment method dynamically.

---

## 🧰 3. Structural Patterns

### Builder Pattern
- Used for creating complex objects step by step.
- Example: Building product cards with optional fields.

---

### Factory Pattern
- Creates objects without exposing the creation logic.
- Used for dynamic creation (e.g., screens, payments)

---

### Singleton Pattern
- Only one instance exists for the entire app.
- Used for API manager, DB handler, Auth token

✅ Useful  
❌ Risky if abused

---

## 📱 4. UI/UX Design Patterns

| Pattern             | Description                                 |
|---------------------|---------------------------------------------|
| Navigation Drawer   | Slide-out menu for navigation               |
| Bottom Navigation   | Quick access to 3–5 main sections           |
| Infinite Scroll     | Load more items as the user scrolls         |
| Pull to Refresh     | Drag to refresh content (used in lists)     |
| Master-Detail View  | Shows a list and the selected item          |

---

## 🧠 Summary Table

| Category         | Pattern        | Common Use Case                    |
|------------------|----------------|------------------------------------|
| Architecture     | MVC, MVVM, MVP | App structure                      |
| Behavioral       | Observer       | Live data updates                  |
| Structural       | Singleton      | Shared instances (API, DB)         |
| UI/UX            | Navigation     | Navigation and screen transitions  |
