/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 7;
constexpr int PHOENIX_VERSION_MINOR = 4;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Human-readable codename for this release (v7.4 = Arthur). */
constexpr char PHOENIX_CODENAME[] = "Arthur";

/* Combined version string used by /api/system/status and logs. */
constexpr char PHOENIX_VERSION_STRING[] = "7.4.0 \"Arthur\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "7.4.0 \"Arthur\""

#endif // PHOENIX_VERSION_HPP
