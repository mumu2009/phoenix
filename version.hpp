/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 8;
constexpr int PHOENIX_VERSION_MINOR = 5;
constexpr int PHOENIX_VERSION_PATCH = 1;

/* Codename rule: ONLY major releases (x.0.0) carry a codename.
   Minor releases (8.x.0) and debug/patch releases (8.5.x) have NO codename.
   8.5.0 is a minor release, so the codename is intentionally empty. */
constexpr char PHOENIX_CODENAME[] = "";

/* Combined version string used by /api/system/status and logs.
   No codename suffix: only x.0.0 releases append " \"Codename\"". */
constexpr char PHOENIX_VERSION_STRING[] = "8.5.1";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "8.5.1"

#endif // PHOENIX_VERSION_HPP
