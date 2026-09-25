/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 9;
constexpr int PHOENIX_VERSION_MINOR = 0;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Codename rule: ONLY major releases (x.0.0) carry a codename.
   Minor releases (8.x.0) and debug/patch releases (8.5.x) have NO codename.
   9.0.0 is a major release: codename Tristan (recovered from the
   withdrawn 8.5 plan; 8.5 shipped without a codename). */
constexpr char PHOENIX_CODENAME[] = "Tristan";

/* Combined version string used by /api/system/status and logs.
   Codename suffix: only x.0.0 releases append " \"Codename\"". */
constexpr char PHOENIX_VERSION_STRING[] = "9.0.0 \"Tristan\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "9.0.0"

#endif // PHOENIX_VERSION_HPP
