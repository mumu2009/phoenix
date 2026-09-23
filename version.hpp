/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 8;
constexpr int PHOENIX_VERSION_MINOR = 5;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Human-readable codename for this release (v8.5 = Tristan). */
constexpr char PHOENIX_CODENAME[] = "Tristan";

/* Combined version string used by /api/system/status and logs. */
constexpr char PHOENIX_VERSION_STRING[] = "8.5.0 \"Tristan\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "8.5.0 \"Tristan\""

#endif // PHOENIX_VERSION_HPP
