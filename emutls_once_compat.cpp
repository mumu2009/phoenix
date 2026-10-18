/* Conan drogon/1.9.5 was built with MinGW emulated TLS, so its objects
   call __emutls_get_address on __emutls_v._ZSt11__once_call. GCC 16's
   libstdc++ keeps those objects behind __get_once_call / __get_once_callable
   and does not export the old emutls names. --wrap sends only those two
   control blocks at the live libstdc++ objects. Other emutls locals still
   use __real___emutls_get_address. */

#include <cstddef>
#include <cstdint>

namespace std {
using phx_once_fn = void (*)();
phx_once_fn &__get_once_call() noexcept;
void *&__get_once_callable() noexcept;
}

struct phx_emutls_object {
  std::uint64_t size;
  std::uint64_t align;
  std::uint64_t offset;
  void *templ;
};

extern "C" {

phx_emutls_object phx_emutls_once_call
    __asm__("__emutls_v._ZSt11__once_call") = {
        sizeof(void *), alignof(void *), 0, nullptr};
phx_emutls_object phx_emutls_once_callable
    __asm__("__emutls_v._ZSt15__once_callable") = {
        sizeof(void *), alignof(void *), 0, nullptr};

void *__real___emutls_get_address(phx_emutls_object *obj);

void *__wrap___emutls_get_address(phx_emutls_object *obj) {
  if (obj == &phx_emutls_once_call) {
    std::phx_once_fn &slot = std::__get_once_call();
    return static_cast<void *>(&slot);
  }
  if (obj == &phx_emutls_once_callable) {
    void *&slot = std::__get_once_callable();
    return static_cast<void *>(&slot);
  }
  return __real___emutls_get_address(obj);
}

}
