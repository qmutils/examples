#pragma once

struct Vec2 {
  float x, y;
  Vec2 operator+(const Vec2& other) { return Vec2{x + other.x, y + other.y}; }
  Vec2 operator*(float a) { return Vec2{x * a, y * a}; }
};
