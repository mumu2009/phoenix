/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 8;
constexpr int PHOENIX_VERSION_MINOR = 0;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Human-readable codename for this release (v8.0 = Lancelot). */
constexpr char PHOENIX_CODENAME[] = "Lancelot";

/* Combined version string used by /api/system/status and logs. */
constexpr char PHOENIX_VERSION_STRING[] = "8.0.0 \"Lancelot\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "8.0.0 \"Lancelot\""

#endif // PHOENIX_VERSION_HPP
