#pragma once

#include <random>

namespace ctranslate2 {

  // Sets the seed of the random generators. It takes effect on every thread at its next random draw,
  // including threads that have already drawn: each generator starts over from this seed.
  void set_random_seed(const unsigned int seed);
  unsigned int get_random_seed();
  // A number that changes each time set_random_seed is called, so a generator can tell it must start over.
  unsigned int get_random_seed_version();
  std::mt19937& get_random_generator();

}
