/* version.hpp - Phoenix version and codename constants */

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 10;
constexpr int PHOENIX_VERSION_MINOR = 0;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Codename rule: ONLY major releases (x.0.0) carry a codename.
   Minor releases (8.x.0) and debug/patch releases (8.5.x) have NO codename.
   10.0.0 is a major release: codename Gareth (the next decree name
   after Tristan). */
constexpr char PHOENIX_CODENAME[] = "Gareth";

/* Combined version string used by /api/system/status and logs.
   Codename suffix: only x.0.0 releases append " \"Codename\"". */
constexpr char PHOENIX_VERSION_STRING[] = "10.0.0 \"Gareth\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "10.0.0"

#endif // PHOENIX_VERSION_HPP
