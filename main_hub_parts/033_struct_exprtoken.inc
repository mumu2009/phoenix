/* Auto-generated from main.cpp
   Segment: struct_ExprToken
   Original lines: 3603-3618 */

struct ExprToken {
  enum class Type {
    Number,
    Var,
    Op,
    Func,
    LParen,
    RParen,
    Comma
  } type{Type::Number};
  std::string text;
  double value{0.0};
  int precedence{0};
  bool rightAssoc{false};
  int arity{0};
};
