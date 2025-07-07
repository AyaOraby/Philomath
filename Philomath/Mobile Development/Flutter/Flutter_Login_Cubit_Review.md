# 📘 Flutter Login with Cubit – Review Notes

This document summarizes all the important concepts covered in our conversation about building a **Flutter login page using Cubit** and clean UI practices.

---

## ✅ 1. `TextEditingController`

- Used to control and access the value of `TextField` or `TextFormField`.
- `.text` gets or sets the input.
- `.clear()` resets the field.

---

## ✅ 2. `Form` and `GlobalKey<FormState>`

- Wrap input fields with `Form` to enable validation.
- Use `GlobalKey<FormState>` to:
  - Validate all fields: `formKey.currentState!.validate()`
  - Reset/save form

---

## ✅ 3. `TextFormField` Validation

```dart
validator: (value) {
  if (value == null || value.isEmpty) return 'Required';
  return null;
}
```

- Validators must return an error message if the input is invalid.

---

## ✅ 4. Custom Button Widget

```dart
class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final bool loading;

  // Build method with loading indicator or text
}
```

- Reusable and style-consistent buttons.
- Can support loading states.

---

## ✅ 5. What is Cubit?

- `Cubit<T>` is a Flutter Bloc class that emits states of type `T`.
- Used for simple state management.
- Cleaner and lighter than full Bloc.

---

## ✅ 6. LoginCubit Structure

```dart
class LoginCubit extends Cubit<LoginStates> {
  LoginCubit() : super(InitialState());

  void login(String email, String password) {
    emit(LoginLoading());
    // logic
  }
}
```

- Inherits from `Cubit<LoginStates>`
- Manages login logic and emits states

---

## ✅ 7. State Classes

```dart
abstract class LoginStates {}
class InitialState extends LoginStates {}
class LoginLoading extends LoginStates {}
class LoginSuccess extends LoginStates {}
class LoginFailure extends LoginStates {
  final String message;
  LoginFailure(this.message);
}
```

- Use `abstract class` as a base for all possible login states.

---

## ✅ 8. BlocConsumer

Used to both **build UI** and **respond to state changes**:

```dart
BlocConsumer<LoginCubit, LoginStates>(
  listener: (context, state) {
    if (state is LoginSuccess) { ... }
    if (state is LoginFailure) { ... }
  },
  builder: (context, state) {
    return CustomButton(...);
  },
)
```

---

## ✅ 9. Code Structure Suggestion

```
lib/
├── cubit/
│   ├── login_cubit.dart
│   └── login_states.dart
├── widgets/
│   └── custom_button.dart
├── screens/
│   └── login_page.dart
```

---

## 🧠 Summary

- Use `Form` and validation for safe input.
- `Cubit` makes login logic clean and separate from UI.
- `BlocConsumer` connects UI with logic and state feedback.
- Custom widgets improve reusability and design.

---

Happy Fluttering! 🚀